#!/usr/bin/env python3

import os
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from rouge_score.rouge_scorer import RougeScorer
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import tiktoken

try:
    import jieba
except ImportError:
    jieba = None

try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception:
    pass

MODEL_API_KEY = "YOUR_API" 
MODEL_BASE_URL = "YOUR_URL"
MODEL_NAME = "YOUR_MODEL_NAME"

INPUT_FILE = "../../Datasets/COD_DD/COD_DD_test.csv"
OUTPUT_DIR = "../../Outputs/COD_DD/deepseek"
RESULT_FILE = os.path.join(OUTPUT_DIR, "deepseek_results_1.json")
TOKEN_USAGE_FILE = os.path.join(OUTPUT_DIR, "token_usage.json")

PATIENT_INDEX_DIR = "../../Datasets/COD_DD/COD_DD_test_index"
PATIENT_DATA_FILE = os.path.join(PATIENT_INDEX_DIR, "patient_data.csv")
PATIENT_EMBEDDING_FILE = os.path.join(PATIENT_INDEX_DIR, "patient_embeddings.npy")
PATIENT_INDEX_FILE = os.path.join(PATIENT_INDEX_DIR, "patient_index.faiss")
PATIENT_MAPPING_FILE = os.path.join(PATIENT_INDEX_DIR, "patient_mapping.json")


class TokenCounter:
    def __init__(self):
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                self.encoder = tiktoken.get_encoding("p50k_base")
            except Exception:
                self.encoder = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text) // 4

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        if not messages:
            return 0
        token_count = 0
        for message in messages:
            token_count += self.count_tokens(message.get("role", ""))
            token_count += self.count_tokens(message.get("content", ""))
            token_count += 4
        token_count += 2
        return token_count


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
        except Exception:
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
                self.bert_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
                os.makedirs(local_model_path, exist_ok=True)
                self.bert_tokenizer.save_pretrained(local_model_path)
                self.bert_model.save_pretrained(local_model_path)
            except Exception:
                pass
        self.jieba = jieba

    def chinese_tokenize(self, text):
        if self.jieba:
            return list(self.jieba.cut(text))
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return list(text)
        return text.split()

    def calculate_rouge_l(self, prediction: str, reference: str) -> float:
        prediction = prediction.strip()
        reference = reference.strip()
        if not prediction or not reference:
            return 0.0
        if self.jieba:
            ref_tokens = list(self.jieba.cut(reference))
            pred_tokens = list(self.jieba.cut(prediction))
        else:
            if any('\u4e00' <= char <= '\u9fff' for char in reference):
                ref_tokens = list(reference)
                pred_tokens = list(prediction)
            else:
                ref_tokens = reference.split()
                pred_tokens = prediction.split()
        try:
            ref_str = ' '.join(ref_tokens)
            pred_str = ' '.join(pred_tokens)
            scores = self.rouge_scorer.score(ref_str, pred_str)
            return scores['rougeL'].fmeasure
        except Exception:
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
            lcs = lcs_length(ref_tokens, pred_tokens)
            precision = lcs / len(pred_tokens) if pred_tokens else 0.0
            recall = lcs / len(ref_tokens) if ref_tokens else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            return f1

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
        max_len = 512
        inputs1 = self.bert_tokenizer(reference, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        inputs2 = self.bert_tokenizer(prediction, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        with torch.no_grad():
            outputs1 = self.bert_model(**inputs1)
            outputs2 = self.bert_model(**inputs2)
            embeddings1 = outputs1.last_hidden_state.mean(dim=1)
            embeddings2 = outputs2.last_hidden_state.mean(dim=1)
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


class PatientRetriever:
    def __init__(self, data_dir=PATIENT_INDEX_DIR):
        self.data_dir = data_dir
        self.model = None
        self.patient_data = None
        self.patient_ids = None
        self.index = None
        self.embedding_dim = None
        mapping_path = os.path.join(self.data_dir, "patient_mapping.json")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        self.patient_ids = mapping["patient_ids"]
        self.embedding_dim = mapping["embedding_dim"]
        data_path = os.path.join(self.data_dir, "patient_data.csv")
        self.patient_data = pd.read_csv(data_path)
        index_path = os.path.join(self.data_dir, "patient_index.faiss")
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def search(self, query_text, top_k=5):
        query_embedding = self.model.encode([query_text])
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.patient_ids):
                continue
            patient_id = self.patient_ids[idx]
            patient_info = self.patient_data[self.patient_data['id'] == patient_id]
            if len(patient_info) > 0:
                patient_row = patient_info.iloc[0].to_dict()
                patient_row['similarity_score'] = float(score)
                results.append(patient_row)
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('similarity_score', ascending=False)
        return results_df


def call_api(
    prompt: str,
    token_counter: TokenCounter,
    token_usage: Dict,
    max_retries: int = 3,
    retry_delay: int = 2,
    api_key: str = ""
) -> Tuple[Optional[str], bool]:
    system_message = (
        "You are an experienced medical professional providing concise, helpful advice to patients.\n"
        "Your responses should:\n"
        "- Be direct and conversational, as if speaking to a patient in a consultation\n"
        "- Focus on clear assessment and practical recommendations\n"
        "- Use natural medical language that balances accuracy with accessibility\n"
        "- Avoid unnecessary formality or verbosity\n"
        "- Never exceed 800 words\n"
        "Remember that the most helpful medical responses are clear, specific, and focused on addressing the patient's immediate concerns."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    input_tokens = token_counter.count_messages_tokens(messages)
    token_usage["total_input_tokens"] += input_tokens
    token_usage["chat_completions"]["input_tokens"] += input_tokens
    token_usage["chat_completions"]["requests"] += 1
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url=MODEL_BASE_URL)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=600,
                stream=False
            )
            result = response.choices[0].message.content.strip()
            output_tokens = token_counter.count_tokens(result)
            token_usage["total_output_tokens"] += output_tokens
            token_usage["chat_completions"]["output_tokens"] += output_tokens
            return result, False
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return None, True


def create_rag_prompt(dialogue: str, related_knowledge: str, similar_patients_info: List[Dict]) -> str:
    similar_patients_text = ""
    for i, patient in enumerate(similar_patients_info, 1):
        disease = patient.get('target_disease', 'Unknown')
        answer = patient.get('answer', '')
        similar_patients_text += f"Case {i}: {disease}\nSample: {answer[:150]}...\n\n"
    knowledge_reference = "Related conditions from knowledge base: " + ", ".join(
        [k.split(":")[0] for k in related_knowledge.split("\n") if k]
    )
    prompt = (
        "⚠️ [CRITICAL: YOUR RESPONSE MUST BE BRIEF - NO MORE THAN 400-600 WORDS TOTAL] ⚠️\n\n"
        "Review this patient dialogue and provide a BRIEF, helpful response similar to those a doctor would give in a real consultation.\n\n"
        f"Patient Dialogue:\n{dialogue}\n\n"
        f"{knowledge_reference}\n\n"
        "After analyzing the dialogue and relevant medical knowledge, compose a concise response that:\n"
        "1. Opens with \"It sounds like you may be experiencing...\" or similar tentative language\n"
        "2. Contains ONLY 2-3 short paragraphs total\n"
        "3. Briefly mentions the most likely condition without excessive certainty\n"
        "4. Provides only the most essential next steps or recommendations\n"
        "5. Uses natural, flowing language without headings, bullet points, or technical jargon\n"
        "6. Maintains a warm but professional tone\n\n"
        "Your entire response must be similar in length and style to these examples:\n"
        "- \"It sounds like you may be experiencing symptoms of [condition]. This can cause [brief explanation]. I recommend [1-2 key recommendations].\"\n"
        "- \"Based on your symptoms, it's possible you have [condition]. This is characterized by [brief explanation]. I suggest [1-2 key recommendations].\"\n\n"
        "CRITICAL: Your response should be approximately 400-600 characters in length (about 100-150 words), similar to what a doctor would say during a brief consultation. Excessive detail will reduce effectiveness.\n\n"
        "Your concise response:"
    )
    return prompt


def save_token_usage(token_usage: Dict, filename: str = TOKEN_USAGE_FILE):
    with open(filename, 'w') as f:
        json.dump(token_usage, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='RAG-based Medical Q&A System')
    parser.add_argument('--input', type=str, default=INPUT_FILE, help='Input CSV file path')
    parser.add_argument('--output', type=str, default=RESULT_FILE, help='Output result file path')
    parser.add_argument('--api-key', type=str, default=MODEL_API_KEY, help='DeepSeek API key')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--top-k', type=int, default=5, help='Number of similar patients to retrieve')
    parser.add_argument('--skip-empty', action='store_true', help='Skip samples without dialogue history')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    token_counter = TokenCounter()
    token_usage = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "chat_completions": {
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0
        }
    }
    df = pd.read_csv(args.input)
    required_columns = ['dialogue', 'answer', 'related_knowledge']
    for column in required_columns:
        if column not in df.columns:
            print(f"Error: Required column '{column}' missing from CSV file")
            return 1
    if args.samples and args.samples > 0:
        df = df.head(args.samples)
    metrics_calculator = MetricsCalculator()
    patient_retriever = PatientRetriever()
    results = []
    stats = {
        "total_processed": 0,
        "successful_generation": 0,
        "failed_generation": 0,
        "skipped_empty_dialogue": 0,
        "api_error_cases": 0,
        "metrics_sum": {
            "rouge_l": 0.0,
            "bleu": 0.0,
            "meteor": 0.0,
            "bert_score": 0.0
        }
    }
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
        dialogue = row.get('dialogue', '')
        reference_answer = row.get('answer', '')
        related_knowledge = row.get('related_knowledge', '')
        if not dialogue and args.skip_empty:
            stats["skipped_empty_dialogue"] += 1
            continue
        stats["total_processed"] += 1
        try:
            similar_patients = patient_retriever.search(dialogue, top_k=args.top_k)
            similar_patients_info = []
            if not similar_patients.empty:
                for _, patient in similar_patients.iterrows():
                    patient_info = {
                        'id': patient.get('id', ''),
                        'similarity_score': patient.get('similarity_score', 0.0),
                        'target_disease': patient.get('target_disease', ''),
                        'answer': patient.get('answer', '')
                    }
                    similar_patients_info.append(patient_info)
            prompt = create_rag_prompt(dialogue, related_knowledge, similar_patients_info)
            generated_answer, api_error = call_api(
                prompt, token_counter, token_usage, api_key=args.api_key
            )
            if api_error or not generated_answer:
                generated_answer = "API_ERROR"
                stats["failed_generation"] += 1
                stats["api_error_cases"] += 1
                result_item = {
                    "id": row.get('id', str(index+1)),
                    "dialogue": dialogue,
                    "reference_answer": reference_answer,
                    "generated_answer": generated_answer,
                    "similar_patients": [p for p in similar_patients_info],
                    "related_knowledge": related_knowledge,
                    "api_error": True,
                    "metrics": {
                        "rouge_l": 0.0,
                        "bleu": 0.0,
                        "meteor": 0.0,
                        "bert_score": 0.0
                    }
                }
                results.append(result_item)
                continue
            stats["successful_generation"] += 1
            metrics = metrics_calculator.calculate_all_metrics(generated_answer, reference_answer)
            for metric_name, value in metrics.items():
                stats["metrics_sum"][metric_name] += value
            result_item = {
                "id": row.get('id', str(index+1)),
                "dialogue": dialogue,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "similar_patients": [p for p in similar_patients_info],
                "related_knowledge": related_knowledge,
                "api_error": False,
                "metrics": metrics
            }
            results.append(result_item)
            if (index + 1) % 10 == 0 or (index + 1) == len(df):
                avg_metrics = {}
                if stats["successful_generation"] > 0:
                    for metric_name, sum_value in stats["metrics_sum"].items():
                        avg_metrics[metric_name] = sum_value / stats["successful_generation"]
                output_data = {
                    "results": results,
                    "statistics": {
                        "total_processed": stats["total_processed"],
                        "successful_generation": stats["successful_generation"],
                        "failed_generation": stats["failed_generation"],
                        "skipped_empty_dialogue": stats["skipped_empty_dialogue"],
                        "api_error_cases": stats["api_error_cases"]
                    },
                    "average_metrics": avg_metrics,
                    "token_usage": {
                        "total_input_tokens": token_usage["total_input_tokens"],
                        "total_output_tokens": token_usage["total_output_tokens"],
                        "total_tokens": token_usage["total_input_tokens"] + token_usage["total_output_tokens"]
                    },
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                save_token_usage(token_usage)
        except Exception as e:
            stats["failed_generation"] += 1
            result_item = {
                "id": row.get('id', str(index+1)),
                "dialogue": dialogue,
                "reference_answer": reference_answer,
                "generated_answer": "ERROR",
                "similar_patients": [],
                "related_knowledge": related_knowledge,
                "error": str(e),
                "api_error": False,
                "metrics": {
                    "rouge_l": 0.0,
                    "bleu": 0.0,
                    "meteor": 0.0,
                    "bert_score": 0.0
                }
            }
            results.append(result_item)
    avg_metrics = {}
    if stats["successful_generation"] > 0:
        for metric_name, sum_value in stats["metrics_sum"].items():
            avg_metrics[metric_name] = sum_value / stats["successful_generation"]
    output_data = {
        "results": results,
        "statistics": {
            "total_processed": stats["total_processed"],
            "successful_generation": stats["successful_generation"],
            "failed_generation": stats["failed_generation"],
            "skipped_empty_dialogue": stats["skipped_empty_dialogue"],
            "api_error_cases": stats["api_error_cases"]
        },
        "average_metrics": avg_metrics,
        "token_usage": {
            "total_input_tokens": token_usage["total_input_tokens"],
            "total_output_tokens": token_usage["total_output_tokens"],
            "total_tokens": token_usage["total_input_tokens"] + token_usage["total_output_tokens"]
        },
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    save_token_usage(token_usage)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())