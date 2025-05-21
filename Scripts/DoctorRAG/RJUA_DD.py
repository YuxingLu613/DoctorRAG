#!/usr/bin/env python3

import json
import os
import time
import numpy as np
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
from openai import OpenAI
import faiss
import pickle
from sentence_transformers import SentenceTransformer

TEST_DATA_PATH = "../../Datasets/RJUA-QA/RJUA_test.json"
DISEASE_FILE_PATH = "../../Datasets/RJUA-QA/disease.txt"
OUTPUT_DIR = "../../Output/RJUA-QA/deepseek"
RESULT_FILE = "../../Output/RJUA-QA/deepseek/rjua_disease_diagnosis_rag_enhanced.json"
PATIENT_INDEX_PATH = "../../Patient_Base/RJUA-QA/RJUA-QA-V1/RJUA_train_faiss.index"
PATIENT_ID_MAP_PATH = "../../Patient_Base/RJUA-QA/RJUA-QA-V1/RJUA_train_id_map.pkl.json"
API_KEY = "YOUR_API_KEY"

def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_faiss_index_and_mapping():
    index = faiss.read_index(PATIENT_INDEX_PATH)
    id_mapping = None
    try:
        with open(PATIENT_ID_MAP_PATH, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
    except json.JSONDecodeError:
        with open(PATIENT_ID_MAP_PATH, 'rb') as f:
            id_mapping = pickle.load(f)
    return index, id_mapping

def load_train_data(id_list):
    data = []
    train_file_path = "../../Patient_Base/RJUA-QA/RJUA_train.json"
    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    if item.get("id") in id_list:
                        data.append(item)
                except json.JSONDecodeError:
                    continue
    return data

def retrieve_similar_patients(question, embedding_model, faiss_index, id_mapping, top_k=3):
    question_embedding = embedding_model.encode([question])
    distances, indices = faiss_index.search(question_embedding, top_k)
    similar_ids = []
    for idx in indices[0]:
        if isinstance(id_mapping, dict):
            similar_ids.append(id_mapping[str(idx)])
        elif isinstance(id_mapping, list) and 0 <= idx < len(id_mapping):
            similar_ids.append(id_mapping[idx])
        else:
            similar_ids.append(str(idx))
    similar_patients = load_train_data(similar_ids)
    return similar_patients

def load_diseases(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        diseases = [line.strip() for line in f.readlines() if line.strip()]
    return diseases

def load_test_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append({
                        "id": item.get("id", ""), 
                        "question": item.get("question", ""),
                        "true_disease": item.get("disease", ""),
                        "context": item.get("context", "")
                    })
                except json.JSONDecodeError:
                    pass
    if max_samples and max_samples > 0:
        data = data[:max_samples]
    return data

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def create_disease_prediction_prompt_with_rag(
    question: str, 
    diseases: List[str], 
    context: str,
    similar_patients: List[Dict[str, Any]]
) -> str:
    diseases_str = "\n".join(diseases)
    similar_patients_info = ""
    if similar_patients:
        similar_patients_info = "## 相似患者案例:\n"
        for i, patient in enumerate(similar_patients):
            patient_question = patient.get("question", "")
            patient_disease = patient.get("disease", "")
            patient_advice = patient.get("advice", "")
            similar_patients_info += f"### 案例 {i+1}:\n"
            similar_patients_info += f"患者描述: {patient_question}\n"
            similar_patients_info += f"诊断: {patient_disease}\n"
            similar_patients_info += f"治疗建议: {patient_advice}\n\n"
    prompt = f"""你是一位专业的泌尿外科医学诊断专家，需要根据患者的描述预测可能的疾病。

                ## 患者的描述:
                {question}

                ## 相关医学知识:
                {context}

                {similar_patients_info}

                请从以下疾病列表中选择最可能的疾病（可以选择1-5种疾病，按可能性从高到低排序）:
                {diseases_str}

                请直接输出疾病名称，多个疾病用逗号分隔，不要有其他解释。例如: "肾结石,肾盂肾炎,尿路感染"
                注意：尽可能全面地考虑各种可能的疾病，不要遗漏。利用相关医学知识和相似患者案例来提高预测准确性。
                """
                    return prompt

                def evaluate_with_llm_agent_strict(
                    predicted_diseases: str,
                    true_disease: str,
                    api_key: str = API_KEY,
                    count_tokens_flag: bool = False
                ) -> Tuple[bool, str, Dict[str, str], Dict[str, int]]:
                    token_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    if not predicted_diseases or not true_disease:
                        return False, "缺少预测或真实疾病信息", {}, token_info
                    prompt = f"""你是一位医学评估专家，需要判断预测的疾病是否全面覆盖了真实疾病。
                    
                预测的疾病: {predicted_diseases}
                真实疾病: {true_disease}

                评估标准(严格): 所有真实疾病都必须在预测疾病中找到对应或相似的疾病，才算准确。
                相似的定义: 两者表达的是同一种或密切相关的医学状况，或者一个是另一个的子类。

                分析步骤:
                1. 将真实疾病拆分为单独的疾病（如果有多个）
                2. 对每个真实疾病，检查预测疾病中是否有对应或相似疾病
                3. 如果所有真实疾病都能找到对应或相似的预测疾病，则评估为"准确"
                4. 如果有任何一个真实疾病在预测中找不到对应或相似疾病，则评估为"不准确"

                请你的回答必须以"准确"或"不准确"开头，然后详细说明每个真实疾病是否在预测中找到对应，如：
                "准确，因为真实疾病A对应预测疾病B，真实疾病C对应预测疾病D..."
                或
                "不准确，因为真实疾病X在预测中找不到对应或相似疾病..."
                """
    result, eval_token_info = call_llm_api(
        prompt, 
        api_key=api_key, 
        count_tokens_flag=count_tokens_flag
    )
    if count_tokens_flag:
        token_info = eval_token_info
    is_accurate = False
    if not result:
        return False, "API调用失败，无法获取评估结果", {}, token_info
    if result.startswith("准确"):
        is_accurate = True
    elif result.startswith("不准确"):
        is_accurate = False
    else:
        lower_response = result.lower()
        if "准确" in lower_response and "不准确" not in lower_response:
            is_accurate = True
        elif "不准确" in lower_response:
            is_accurate = False
        else:
            is_accurate = False
    matches = {}
    true_diseases = [d.strip() for d in true_disease.split(',') if d.strip()]
    for td in true_diseases:
        if is_accurate:
            matches[td] = "找到匹配或相似疾病"
        else:
            matches[td] = "未找到匹配或相似疾病"
    for td in true_diseases:
        for segment in result.split('，'):
            if td in segment and "对应" in segment:
                matches[td] = segment.strip()
                break
    return is_accurate, result, matches, token_info

def call_llm_api(
    prompt: str,
    max_retries: int = 3,
    retry_delay: int = 2,
    api_key: str = API_KEY,
    count_tokens_flag: bool = False
) -> Tuple[str, Dict[str, int]]:
    token_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            messages = [
                {"role": "system", "content": "你是一个专业的医学诊断助手，擅长根据症状判断疾病"},
                {"role": "user", "content": prompt}
            ]
            if count_tokens_flag:
                system_tokens = count_tokens(messages[0]["content"])
                user_tokens = count_tokens(messages[1]["content"])
                token_info["input_tokens"] = system_tokens + user_tokens
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.1,
                max_tokens=200,
                stream=False
            )
            result = response.choices[0].message.content.strip()
            if count_tokens_flag:
                token_info["output_tokens"] = count_tokens(result)
                token_info["total_tokens"] = token_info["input_tokens"] + token_info["output_tokens"]
            return result, token_info
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return "", token_info

def main():
    parser = argparse.ArgumentParser(description='Predict diseases from patient questions and evaluate accuracy (RAG enhanced)')
    parser.add_argument('--data', type=str, default=TEST_DATA_PATH, help='Test data file path')
    parser.add_argument('--diseases', type=str, default=DISEASE_FILE_PATH, help='Disease list file path')
    parser.add_argument('--output', type=str, default=RESULT_FILE, help='Output result file path')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--api-key', type=str, default=API_KEY, help='API key')
    parser.add_argument('--continue-from', type=str, help='Continue from existing result file')
    parser.add_argument('--similar-patients', type=int, default=3, help='Number of similar patients to retrieve')
    parser.add_argument('--token-stats', action='store_true', help='Whether to collect token usage statistics')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = []
    accuracy_stats = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "no_prediction": 0,
        "accuracy": 0.0,
        "failed_calls": 0
    }
    token_stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "avg_input_tokens": 0,
        "avg_output_tokens": 0
    }
    start_index = 0
    if args.continue_from and os.path.exists(args.continue_from):
        with open(args.continue_from, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get("results", [])
            accuracy_stats = data.get("accuracy_stats", accuracy_stats)
            start_index = len(results)
    diseases = load_diseases(args.diseases)
    test_data = load_test_data(args.data, args.samples)
    if start_index > 0:
        test_data = test_data[start_index:]
    embedding_model = load_embedding_model()
    faiss_index, id_mapping = load_faiss_index_and_mapping()
    for i, item in enumerate(tqdm(test_data, desc="Processing data")):
        current_index = start_index + i
        item_id = item.get("id", str(current_index+1))
        question = item.get("question", "")
        true_disease = item.get("true_disease", "")
        context = item.get("context", "")
        try:
            similar_patients = []
            if embedding_model is not None and faiss_index is not None and id_mapping is not None:
                similar_patients = retrieve_similar_patients(
                    question, 
                    embedding_model, 
                    faiss_index, 
                    id_mapping, 
                    top_k=args.similar_patients
                )
            prompt = create_disease_prediction_prompt_with_rag(
                question,
                diseases,
                context,
                similar_patients
            )
            predicted_diseases, pred_token_info = call_llm_api(
                prompt, 
                api_key=args.api_key,
                count_tokens_flag=args.token_stats
            )
            if args.token_stats:
                token_stats["total_input_tokens"] += pred_token_info["input_tokens"]
                token_stats["total_output_tokens"] += pred_token_info["output_tokens"]
                token_stats["total_tokens"] += pred_token_info["total_tokens"]
            if not predicted_diseases:
                accuracy_stats["failed_calls"] += 1
                results.append({
                    "id": item_id,
                    "question": question,
                    "true_disease": true_disease,
                    "predicted_diseases": "FAILED_API_CALL",
                    "is_accurate": None,
                    "evaluation_reason": "API调用失败",
                    "matches_info": {},
                    "similar_patients_count": len(similar_patients),
                    "token_info": pred_token_info if args.token_stats else {}
                })
                continue
            is_accurate, evaluation_reason, matches_info, eval_token_info = evaluate_with_llm_agent_strict(
                predicted_diseases, 
                true_disease, 
                api_key=args.api_key,
                count_tokens_flag=args.token_stats
            )
            if args.token_stats:
                token_stats["total_input_tokens"] += eval_token_info["input_tokens"]
                token_stats["total_output_tokens"] += eval_token_info["output_tokens"]
                token_stats["total_tokens"] += eval_token_info["total_tokens"]
            if evaluation_reason != "API调用失败":
                accuracy_stats["total"] += 1
                if is_accurate:
                    accuracy_stats["correct"] += 1
                else:
                    accuracy_stats["incorrect"] += 1
                if accuracy_stats["total"] > 0:
                    accuracy_stats["accuracy"] = accuracy_stats["correct"] / accuracy_stats["total"]
                if args.token_stats and accuracy_stats["total"] > 0:
                    token_stats["avg_input_tokens"] = token_stats["total_input_tokens"] / (accuracy_stats["total"] + accuracy_stats["failed_calls"])
                    token_stats["avg_output_tokens"] = token_stats["total_output_tokens"] / (accuracy_stats["total"] + accuracy_stats["failed_calls"])
            if evaluation_reason != "API调用失败":
                results.append({
                    "id": item_id,
                    "question": question,
                    "true_disease": true_disease,
                    "predicted_diseases": predicted_diseases,
                    "is_accurate": is_accurate,
                    "evaluation_reason": evaluation_reason,
                    "matches_info": matches_info,
                    "similar_patients_count": len(similar_patients),
                    "token_info": {
                        "pred_tokens": pred_token_info,
                        "eval_tokens": eval_token_info
                    } if args.token_stats else {}
                })
        except Exception as e:
            accuracy_stats["failed_calls"] += 1
            results.append({
                "id": item_id,
                "question": question,
                "true_disease": true_disease,
                "predicted_diseases": "ERROR: " + str(e),
                "is_accurate": None,
                "error": str(e),
                "similar_patients_count": len(similar_patients) if 'similar_patients' in locals() else 0
            })
        if (i + 1) % 10 == 0 or (i + 1) == len(test_data):
            with open(args.output, 'w', encoding='utf-8') as f:
                output_data = {
                    "results": results,
                    "accuracy_stats": accuracy_stats,
                    "total_samples": start_index + len(test_data),
                    "processed_samples": start_index + i + 1,
                    "valid_samples": accuracy_stats["total"],
                    "failed_calls": accuracy_stats["failed_calls"],
                    "evaluation_standard": "Strict standard - all true diseases must be found in prediction",
                    "evaluation_method": "LLM agent",
                    "enhancement": "RAG (knowledge base retrieval + patient similarity retrieval)"
                }
                if args.token_stats:
                    output_data["token_stats"] = token_stats
                json.dump(output_data, f, ensure_ascii=False, indent=2)
    disease_stats = {}
    for result in results:
        if result.get("is_accurate") is None:
            continue
        true_disease = result.get("true_disease", "")
        if true_disease:
            for disease in true_disease.split(","):
                disease = disease.strip()
                if disease:
                    if disease not in disease_stats:
                        disease_stats[disease] = {"total": 0, "correct": 0, "accuracy": 0.0}
                    disease_stats[disease]["total"] += 1
                    if result.get("is_accurate", False):
                        disease_stats[disease]["correct"] += 1
    for disease, stats in disease_stats.items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
    with open(args.output, 'w', encoding='utf-8') as f:
        output_data = {
            "results": results,
            "accuracy_stats": accuracy_stats,
            "disease_stats": disease_stats,
            "total_samples": start_index + len(test_data),
            "processed_samples": start_index + len(test_data),
            "valid_samples": accuracy_stats["total"],
            "failed_calls": accuracy_stats["failed_calls"],
            "evaluation_standard": "Strict standard - all true diseases must be found in prediction",
            "evaluation_method": "LLM agent",
            "enhancement": "RAG (knowledge base retrieval + patient similarity retrieval)"
        }
        if args.token_stats:
            output_data["token_stats"] = token_stats
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    return 0

if __name__ == "__main__":
    main()