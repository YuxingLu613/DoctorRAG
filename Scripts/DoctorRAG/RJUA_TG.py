#!/usr/bin/env python3

import json
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import argparse
from typing import List, Dict, Any, Tuple, Optional
import re
from rouge_score.rouge_scorer import RougeScorer
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
import tiktoken

try:
    import jieba
except ImportError:
    jieba = None

try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

TEST_DATA_PATH = "../../Datasets/RJUA-QA/RJUA_test.json"
DISEASE_FILE_PATH = "../../Datasets/RJUA-QA/disease.txt"
ADVICE_FILE_PATH = "../../Datasets/RJUA-QA/advice.txt"
OUTPUT_DIR = "../../Output/RJUA-QA/deepseek"
RESULT_FILE = "../../Output/RJUA-QA/deepseek/rjua_disease_diagnosis_rag_enhanced.json"
PATIENT_INDEX_PATH = "../../Patient_Base/RJUA-QA/RJUA-QA-V1/RJUA_train_faiss.index"
PATIENT_ID_MAP_PATH = "../../Patient_Base/RJUA-QA/RJUA-QA-V1/RJUA_train_id_map.pkl.json"
TRAIN_DATA_FILE = "../../Patient_Base/RJUA-QA/RJUA_train.json"
API_KEY = "YOUR_API_KEY"

class PatientRetriever:
    def __init__(self, index_file: str, id_map_file: str, train_data_file: str, 
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.index = faiss.read_index(index_file)
        with open(id_map_file, 'rb') as f:
            self.id_map = pickle.load(f)
        with open(train_data_file, 'r', encoding='utf-8') as f:
            self.train_data = self.load_json_safe(f)
            self.id_to_data = {item['id']: item for item in self.train_data}
        self.model = SentenceTransformer(model_name)
    
    def load_json_safe(self, file_obj):
        try:
            return json.load(file_obj)
        except json.JSONDecodeError:
            file_obj.seek(0)
            content = file_obj.read()
            lines = content.strip().split('\n')
            results = []
            for line in lines:
                try:
                    item = json.loads(line)
                    results.append(item)
                except:
                    continue
            return results
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)
        retrieved_patients = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.id_map):
                patient_id = self.id_map[idx]
                if patient_id in self.id_to_data:
                    patient_data = self.id_to_data[patient_id].copy()
                    patient_data['similarity_score'] = float(score)
                    retrieved_patients.append(patient_data)
        return retrieved_patients

def load_test_dataset(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    data = []
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                if not data:
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, list):
                                data = value
                                break
            except:
                pass
    if not isinstance(data, list):
        if isinstance(data, dict):
            data = [data]
        else:
            data = []
    if max_samples and max_samples > 0:
        data = data[:max_samples]
    return data

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def call_llm_api(prompt: str, max_retries: int = 3, retry_delay: int = 2, api_key: str = API_KEY) -> Tuple[Optional[str], Dict[str, int]]:
    token_stats = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    token_stats["input_tokens"] = count_tokens(prompt)
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                stream=False
            )
            result = response.choices[0].message.content.strip()
            token_stats["output_tokens"] = count_tokens(result)
            token_stats["total_tokens"] = token_stats["input_tokens"] + token_stats["output_tokens"]
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                if hasattr(usage, 'prompt_tokens'):
                    token_stats["input_tokens"] = usage.prompt_tokens
                if hasattr(usage, 'completion_tokens'):
                    token_stats["output_tokens"] = usage.completion_tokens
                if hasattr(usage, 'total_tokens'):
                    token_stats["total_tokens"] = usage.total_tokens
            return result, token_stats
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return None, token_stats

def create_medical_qa_prompt(question: str, context: str, retrieved_cases: List[Dict[str, Any]], 
                             disease_file: str = DISEASE_FILE_PATH, advice_file: str = ADVICE_FILE_PATH) -> str:
    case_context = ""
    for i, case in enumerate(retrieved_cases[:3], 1):
        case_context += f"[Similar Case {i}]\n"
        case_context += f"Symptom Description: {case.get('question', '')}\n"
        case_context += f"Reference Answer: {case.get('answer', '')}\n"
        case_context += f"Similarity Score: {case.get('similarity_score', 0):.3f}\n\n"
    prompt = (
        f"You are a professional medical consultant responsible for answering patient health questions. "
        f"Please provide a concise and accurate answer based on the provided knowledge and similar cases.\n\n"
        f"Strictly follow the answer format below:\n"
        f"1. Start with 'Hello, based on your symptom description'\n"
        f"2. After referring to retrieved similar cases and relevant medical knowledge, clearly state the patient's disease, which must come from {disease_file}\n"
        f"3. After referring to retrieved similar cases and relevant medical knowledge, provide diagnostic advice, which must come from {advice_file}\n"
        f"4. Do not repeat information already provided by the patient, and do not mention which similar cases or knowledge you referenced\n"
        f"5. You may list several diseases and diagnostic suggestions, keep the answer around 200 words, not too short, and not too long\n"
        f"6. Please answer in a single coherent paragraph, do not use line breaks or numbered lists, separate multiple diseases and suggestions with commas\n\n"
        f"Patient Question:\n{question}\n\n"
        f"Retrieved Similar Cases:\n{case_context}\n"
        f"Relevant Medical Knowledge:\n{context}\n\n"
        f"Please provide your answer:"
    )
    return prompt

class MetricsCalculator:
    def __init__(self):
        self.rouge_scorer = RougeScorer(['rougeL'], use_stemmer=False)
        self.smooth_bleu = SmoothingFunction().method1
        local_model_path = "./models/chinese-roberta-wwm-ext"
        self.bert_tokenizer = None
        self.bert_model = None
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.bert_model = AutoModel.from_pretrained(local_model_path)
        except:
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
                self.bert_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
                os.makedirs(local_model_path, exist_ok=True)
                self.bert_tokenizer.save_pretrained(local_model_path)
                self.bert_model.save_pretrained(local_model_path)
            except:
                pass
        try:
            import jieba
            self.jieba = jieba
        except ImportError:
            self.jieba = None
    
    def chinese_tokenize(self, text):
        if self.jieba:
            return list(self.jieba.cut(text))
        else:
            return list(text)
    
    def calculate_rouge_l(self, prediction: str, reference: str) -> float:
        try:
            prediction = prediction.strip()
            reference = reference.strip()
            if not prediction or not reference:
                return 0.0
            def lcs_length(a, b):
                m, n = len(a), len(b)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if a[i-1] == b[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                return dp[m][n]
            ref_tokens = self.chinese_tokenize(reference)
            pred_tokens = self.chinese_tokenize(prediction)
            lcs = lcs_length(ref_tokens, pred_tokens)
            precision = lcs / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return f1
        except Exception:
            return 0.0
        
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        if not prediction or not reference:
            return 0.0
        reference_tokens = self.chinese_tokenize(reference)
        prediction_tokens = self.chinese_tokenize(prediction)
        if not reference_tokens or not prediction_tokens:
            return 0.0
        score = sentence_bleu([reference_tokens], prediction_tokens, 
                              smoothing_function=self.smooth_bleu,
                              weights=(0.25, 0.25, 0.25, 0.25))
        return score
    
    def calculate_meteor(self, prediction: str, reference: str) -> float:
        if not prediction or not reference:
            return 0.0
        reference_tokens = self.chinese_tokenize(reference)
        prediction_tokens = self.chinese_tokenize(prediction)
        if not reference_tokens or not prediction_tokens:
            return 0.0
        score = meteor_score([reference_tokens], prediction_tokens)
        return score
    
    def calculate_bert_score(self, prediction: str, reference: str) -> float:
        if self.bert_tokenizer is None or self.bert_model is None:
            return 0.0
        if not prediction or not reference:
            return 0.0
        inputs1 = self.bert_tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
        inputs2 = self.bert_tokenizer(prediction, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings1 = self.bert_model(**inputs1).last_hidden_state.mean(dim=1)
            embeddings2 = self.bert_model(**inputs2).last_hidden_state.mean(dim=1)
        embeddings1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        cosine_sim = torch.mm(embeddings1_norm, embeddings2_norm.transpose(0, 1))
        score = cosine_sim.item()
        return score
    
    def calculate_all_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        prediction = prediction.strip()
        reference = reference.strip()
        rouge_l = self.calculate_rouge_l(prediction, reference)
        bleu = self.calculate_bleu(prediction, reference)
        meteor = self.calculate_meteor(prediction, reference)
        bert_score = self.calculate_bert_score(prediction, reference)
        metrics = {
            "rouge_l": rouge_l,
            "bleu": bleu,
            "meteor": meteor,
            "bert_score": bert_score
        }
        return metrics

def create_medical_qa_prompt_original(question: str, context: str, disease_file: str = DISEASE_FILE_PATH, advice_file: str = ADVICE_FILE_PATH) -> str:
    prompt = (
        f"You are a professional medical consultant responsible for answering patient health questions. "
        f"Please provide a concise and accurate answer based on the provided knowledge.\n\n"
        f"Strictly follow the answer format below:\n"
        f"1. Start with 'Hello, based on your symptom description'\n"
        f"2. Clearly state the patient's disease, which must come from {disease_file}\n"
        f"3. Provide diagnostic advice, which must come from {advice_file}\n"
        f"4. Do not repeat information already provided by the patient\n"
        f"5. You may list several diseases and diagnostic suggestions, keep the answer around 200 words, not too short, and not too long\n\n"
        f"Patient Question:\n{question}\n\n"
        f"Relevant Medical Knowledge:\n{context}\n\n"
        f"Please provide your answer:"
    )
    return prompt

def main():
    parser = argparse.ArgumentParser(description='RAG-based Medical QA Consistency Evaluation')
    parser.add_argument('--data', type=str, help='Test dataset file path')
    parser.add_argument('--samples', type=int, help='Limit number of samples to process')
    parser.add_argument('--api-key', type=str, default=API_KEY, help='LLM API key')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no-retrieval', action='store_true', help='Disable retrieval augmentation')
    args = parser.parse_args()
    global RESULT_FILE
    if args.output:
        RESULT_FILE = args.output
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    retriever = None
    if not args.no_retrieval:
        try:
            retriever = PatientRetriever(PATIENT_INDEX_PATH, PATIENT_ID_MAP_PATH, TRAIN_DATA_FILE)
        except:
            pass
    test_data_path = args.data if args.data else input("Please enter the test dataset file path: ")
    test_dataset = load_test_dataset(test_data_path, args.samples)
    if not test_dataset:
        return 1
    metrics_calculator = MetricsCalculator()
    results = []
    total_token_stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0
    }
    for i, item in enumerate(tqdm(test_dataset, desc="Processing test data")):
        question = item.get("question", "")
        context = item.get("context", "")
        reference_answer = item.get("answer", "")
        try:
            retrieved_cases = []
            if retriever:
                retrieved_cases = retriever.search(question, k=5)
            if retriever:
                prompt = create_medical_qa_prompt(question, context, retrieved_cases)
            else:
                prompt = create_medical_qa_prompt_original(question, context)
            generated_answer, token_stats = call_llm_api(prompt, api_key=args.api_key)
            if not generated_answer:
                generated_answer = ""
            total_token_stats["total_input_tokens"] += token_stats["input_tokens"]
            total_token_stats["total_output_tokens"] += token_stats["output_tokens"]
            total_token_stats["total_tokens"] += token_stats["total_tokens"]
            metrics = metrics_calculator.calculate_all_metrics(generated_answer, reference_answer)
            result_item = {
                "id": item.get("id", str(i+1)),
                "question": question,
                "context": context,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "retrieved_cases": retrieved_cases,
                "metrics": metrics,
                "token_stats": token_stats
            }
            results.append(result_item)
        except Exception as e:
            results.append({
                "id": item.get("id", str(i+1)),
                "question": question,
                "context": context,
                "reference_answer": reference_answer,
                "generated_answer": "ERROR",
                "error": str(e),
                "metrics": {
                    "rouge_l": 0.0,
                    "bleu": 0.0,
                    "meteor": 0.0,
                    "bert_score": 0.0
                },
                "token_stats": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            })
        if (i + 1) % 10 == 0 or (i + 1) == len(test_dataset):
            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "results": results,
                    "total_token_stats": total_token_stats,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, ensure_ascii=False, indent=2)
    metrics_sum = {
        "rouge_l": 0.0,
        "bleu": 0.0,
        "meteor": 0.0,
        "bert_score": 0.0
    }
    valid_count = 0
    for result in results:
        if "metrics" in result and "rouge_l" in result["metrics"]:
            metrics_sum["rouge_l"] += result["metrics"]["rouge_l"]
            metrics_sum["bleu"] += result["metrics"]["bleu"]
            metrics_sum["meteor"] += result["metrics"]["meteor"]
            metrics_sum["bert_score"] += result["metrics"]["bert_score"]
            valid_count += 1
    avg_metrics = {
        "rouge_l": metrics_sum["rouge_l"] / valid_count if valid_count > 0 else 0.0,
        "bleu": metrics_sum["bleu"] / valid_count if valid_count > 0 else 0.0,
        "meteor": metrics_sum["meteor"] / valid_count if valid_count > 0 else 0.0,
        "bert_score": metrics_sum["bert_score"] / valid_count if valid_count > 0 else 0.0
    }
    final_output = {
        "results": results,
        "avg_metrics": avg_metrics,
        "total_samples": len(test_dataset),
        "processed_samples": valid_count,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "rag_enabled": retriever is not None,
        "total_token_stats": total_token_stats,
        "avg_tokens_per_query": {
            "avg_input_tokens": total_token_stats["total_input_tokens"] / valid_count if valid_count > 0 else 0,
            "avg_output_tokens": total_token_stats["total_output_tokens"] / valid_count if valid_count > 0 else 0,
            "avg_total_tokens": total_token_stats["total_tokens"] / valid_count if valid_count > 0 else 0
        }
    }
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    return 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend(['--data', TEST_DATA_PATH])
    sys.exit(main())