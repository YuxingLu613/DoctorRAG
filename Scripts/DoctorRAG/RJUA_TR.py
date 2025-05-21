#!/usr/bin/env python3

import json
import os
import time
import numpy as np
from tqdm import tqdm
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
import tiktoken
from openai import OpenAI
import faiss
import pickle
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TEST_DATA_PATH = "../../Datasets/RJUA-QA/RJUA_test.json"
DISEASE_FILE_PATH = "../../Datasets/RJUA-QA/disease.txt"
ADVICE_FILE_PATH = "../../Datasets/RJUA-QA/advice.txt"
OUTPUT_DIR = "../../Output/RJUA-QA/deepseek"
RESULT_FILE = "../../Output/RJUA-QA/deepseek/rjua_disease_diagnosis_rag_enhanced.json"
PATIENT_INDEX_PATH = "../../Patient_Base/RJUA-QA/RJUA-QA-V1/RJUA_train_faiss.index"
PATIENT_ID_MAP_PATH = "../../Patient_Base/RJUA-QA/RJUA-QA-V1/RJUA_train_id_map.pkl.json"
TRAIN_DATA_FILE = "../../Patient_Base/RJUA-QA/RJUA_train.json"
API_KEY = "YOUR_API_KEY"

def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_faiss_index_and_mapping():
    index = faiss.read_index(PATIENT_INDEX_PATH)
    with open(PATIENT_ID_MAP_PATH, 'r', encoding='utf-8') as f:
        try:
            id_mapping = json.load(f)
        except json.JSONDecodeError:
            with open(PATIENT_ID_MAP_PATH, 'rb') as f2:
                id_mapping = pickle.load(f2)
    return index, id_mapping

def load_train_data(id_list):
    data = []
    with open(TRAIN_DATA_FILE, 'r', encoding='utf-8') as f:
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

def load_advice_list(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

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
                        "true_advice": item.get("advice", ""),
                        "context": item.get("context", "")
                    })
                except json.JSONDecodeError:
                    pass
    if max_samples:
        data = data[:max_samples]
    return data

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def create_advice_prediction_prompt_with_rag(
    question: str,
    advice_list: List[str],
    context: str,
    similar_patients: List[Dict[str, Any]]
) -> str:
    advice_str = "\n".join(advice_list)
    similar_patients_info = ""
    if similar_patients:
        similar_patients_info = "## 相似患者案例:\n"
        for i, patient in enumerate(similar_patients):
            patient_question = patient.get("question", "")
            patient_disease = patient.get("disease", "")
            patient_advice = patient.get("advice", "")
            similar_patients_info += f"### 案例 {i+1}:\n"
            similar_patients_info += f"患者描述: {patient_question}\n"
            if patient_disease:
                similar_patients_info += f"诊断: {patient_disease}\n"
            if patient_advice:
                similar_patients_info += f"治疗建议: {patient_advice}\n"
            similar_patients_info += "\n"
    prompt = f"""你是一位专业的泌尿外科医学诊断专家，需要根据患者的描述预测可能的治疗建议。

                ## 患者的描述:
                {question}

                ## 相关医学知识:
                {context}

                {similar_patients_info}

                请从以下治疗建议列表中选择最合适的治疗建议（可以选择1-5种建议，注意不要超过5条，按重要性从高到低排序）:
                {advice_str}

                请直接输出治疗建议，多个建议用逗号分隔，不要有其他解释。例如: "双侧输尿管支架置入术,抗感染治疗,补液支持治疗"
                注意：请尽可能全面地考虑各种可能的治疗建议，不要遗漏任何重要的治疗建议，但是请一定要注意不要超过5条！
                利用相关医学知识和相似患者案例来提高预测准确性。
                """
                    return prompt

                def create_balanced_advice_evaluation_prompt(predicted_advice: str, true_advice: str) -> str:
                    prompt = f"""你是一位医学评估专家，需要判断预测的治疗建议是否覆盖了真实治疗建议。
                    
                预测的治疗建议: {predicted_advice}
                真实治疗建议: {true_advice}

                评估标准(平衡):
                - 当真实建议有1-2条时，需要全部匹配才算准确
                - 当真实建议大于2条时，只要匹配2条或以上就算准确
                相似的定义: 两者表达的是同一种或密切相关的治疗方法，或者一个是另一个的子类。

                分析步骤:
                1. 将真实治疗建议拆分为单独的建议（如果有多个）
                2. 计算真实建议的总数量
                3. 对每个真实建议，检查预测建议中是否有对应或相似建议
                4. 计算匹配的建议数量
                5. 如果真实建议数量≤2，则需全部匹配才算准确
                6. 如果真实建议数量>2，则匹配数量≥2就算准确

                请你的回答必须以"准确"或"不准确"开头，然后详细说明每个真实建议是否在预测中找到对应，如：
                "准确，因为真实建议A对应预测建议B，真实建议C对应预测建议D..."
                或
                "不准确，因为真实建议X在预测中找不到对应或相似建议..."

                最后请总结匹配数量，例如："总计4条真实建议中有3条找到匹配，符合平衡评估标准（>2条时匹配≥2即可）"。
                """
    return prompt

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
                {"role": "system", "content": "你是一个专业的医学诊断助手，擅长根据症状判断治疗方案"},
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
                max_tokens=400,
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

def evaluate_with_llm_agent_balanced(
    predicted_advice: str,
    true_advice: str,
    api_key: str = API_KEY,
    count_tokens_flag: bool = False
) -> Tuple[bool, str, Dict[str, str], int, int, Dict[str, int]]:
    token_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not predicted_advice or not true_advice:
        return False, "Missing predicted or true advice information", {}, 0, 0, token_info
    prompt = create_balanced_advice_evaluation_prompt(predicted_advice, true_advice)
    response, eval_token_info = call_llm_api(prompt, api_key=api_key, count_tokens_flag=count_tokens_flag)
    if count_tokens_flag:
        token_info = eval_token_info
    is_accurate = False
    if response.startswith("准确"):
        is_accurate = True
    elif response.startswith("不准确"):
        is_accurate = False
    else:
        lower_response = response.lower()
        if "准确" in lower_response and "不准确" not in lower_response:
            is_accurate = True
        elif "不准确" in lower_response:
            is_accurate = False
        else:
            is_accurate = False
    matches = {}
    true_advice_items = [item.strip() for item in true_advice.split(',') if item.strip()]
    matched_count = 0
    for advice in true_advice_items:
        found = False
        if is_accurate:
            for segment in response.split('，'):
                if advice in segment and "对应" in segment:
                    matches[advice] = segment.strip()
                    found = True
                    matched_count += 1
                    break
            if not found:
                if "所有" in response.lower() and "找到匹配" in response.lower():
                    matches[advice] = "找到匹配或相似建议"
                    matched_count += 1
                else:
                    matches[advice] = "未明确说明匹配情况"
        else:
            for segment in response.split('，'):
                if advice in segment and "找不到" in segment:
                    matches[advice] = segment.strip()
                    found = True
                    break
            if not found:
                matches[advice] = "未找到匹配或相似建议"
    if matched_count == 0 and is_accurate:
        matched_count = response.count("对应") + response.count("匹配")
        if matched_count == 0:
            matched_count = min(2, len(true_advice_items))
    if is_accurate:
        if len(true_advice_items) <= 2:
            matched_count = len(true_advice_items)
        else:
            matched_count = max(matched_count, 2)
    return is_accurate, response, matches, len(true_advice_items), matched_count, token_info

def evaluate_prediction_balanced(
    predicted_advice: str,
    true_advice: str
) -> Tuple[bool, Dict[str, bool], int, int]:
    if not predicted_advice or not true_advice:
        return False, {}, 0, 0
    pred_advice_items = [item.strip() for item in predicted_advice.split(',') if item.strip()]
    true_advice_items = [item.strip() for item in true_advice.split(',') if item.strip()]
    matches = {}
    match_count = 0
    for true_item in true_advice_items:
        is_match = False
        for pred_item in pred_advice_items:
            if true_item == pred_item:
                is_match = True
                match_count += 1
                break
            elif true_item in pred_item or pred_item in true_item:
                is_match = True
                match_count += 1
                break
        matches[true_item] = is_match
    true_count = len(true_advice_items)
    if true_count <= 2:
        is_accurate = all(matches.values())
    else:
        is_accurate = match_count >= 2
    return is_accurate, matches, true_count, match_count

def main():
    parser = argparse.ArgumentParser(
        description='Predict treatment advice for patient questions and evaluate accuracy (RAG enhanced)'
    )
    parser.add_argument('--data', type=str, default=TEST_DATA_PATH, help='Test data file path')
    parser.add_argument('--advice', type=str, default=ADVICE_FILE_PATH, help='Advice list file path')
    parser.add_argument('--output', type=str, default=RESULT_FILE, help='Output result file path')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--api-key', type=str, default=API_KEY, help='API key')
    parser.add_argument('--rule-based', action='store_true', help='Use rule-based evaluation instead of LLM agent')
    parser.add_argument('--continue-from', type=str, help='Continue from existing result file')
    parser.add_argument('--similar-patients', type=int, default=3, help='Number of similar patients to retrieve')
    parser.add_argument('--token-stats', action='store_true', help='Collect token usage statistics')
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
    advice_list = load_advice_list(args.advice)
    test_data = load_test_data(args.data, args.samples)
    if start_index > 0:
        test_data = test_data[start_index:]
    embedding_model = load_embedding_model()
    faiss_index, id_mapping = load_faiss_index_and_mapping()
    for i, item in enumerate(tqdm(test_data, desc="Processing data")):
        current_index = start_index + i
        item_id = item.get("id", str(current_index + 1))
        question = item.get("question", "")
        true_advice = item.get("true_advice", "")
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
            prompt = create_advice_prediction_prompt_with_rag(
                question,
                advice_list,
                context,
                similar_patients
            )
            predicted_advice, pred_token_info = call_llm_api(
                prompt,
                api_key=args.api_key,
                count_tokens_flag=args.token_stats
            )
            if args.token_stats:
                token_stats["total_input_tokens"] += pred_token_info["input_tokens"]
                token_stats["total_output_tokens"] += pred_token_info["output_tokens"]
                token_stats["total_tokens"] += pred_token_info["total_tokens"]
            if not predicted_advice:
                accuracy_stats["failed_calls"] += 1
                accuracy_stats["no_prediction"] += 1
                results.append({
                    "id": item_id,
                    "question": question,
                    "true_advice": true_advice,
                    "predicted_advice": "FAILED_API_CALL",
                    "is_accurate": False,
                    "evaluation_reason": "API call failed",
                    "matches_info": {},
                    "similar_patients_count": len(similar_patients),
                    "token_info": pred_token_info if args.token_stats else {}
                })
                continue
            if args.rule_based:
                is_accurate, match_details, true_count, match_count = evaluate_prediction_balanced(
                    predicted_advice, true_advice
                )
                evaluation_reason = f"Rule-based balanced evaluation: {match_count} out of {true_count} true advice matched"
                matches_info = {k: "Matched" if v else "Not matched" for k, v in match_details.items()}
                eval_token_info = {}
            else:
                is_accurate, evaluation_reason, matches_info, true_count, match_count, eval_token_info = evaluate_with_llm_agent_balanced(
                    predicted_advice,
                    true_advice,
                    api_key=args.api_key,
                    count_tokens_flag=args.token_stats
                )
                if args.token_stats:
                    token_stats["total_input_tokens"] += eval_token_info["input_tokens"]
                    token_stats["total_output_tokens"] += eval_token_info["output_tokens"]
                    token_stats["total_tokens"] += eval_token_info["total_tokens"]
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
            results.append({
                "id": item_id,
                "question": question,
                "true_advice": true_advice,
                "predicted_advice": predicted_advice,
                "is_accurate": is_accurate,
                "true_count": true_count,
                "match_count": match_count,
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
            accuracy_stats["total"] += 1
            accuracy_stats["no_prediction"] += 1
            results.append({
                "id": item_id,
                "question": question,
                "true_advice": true_advice,
                "predicted_advice": "ERROR: " + str(e),
                "is_accurate": False,
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
                    "evaluation_standard": "Balanced standard - 1-2 advice must all match, >2 advice at least 2 must match",
                    "evaluation_method": "LLM agent" if not args.rule_based else "Rule-based",
                    "enhancement": "RAG (knowledge base retrieval + patient similarity retrieval)"
                }
                if args.token_stats:
                    output_data["token_stats"] = token_stats
                json.dump(output_data, f, ensure_ascii=False, indent=2)
    advice_stats = {}
    for result in results:
        if result.get("predicted_advice") == "FAILED_API_CALL" or result.get("predicted_advice", "").startswith("ERROR:"):
            continue
        true_advice = result.get("true_advice", "")
        if true_advice:
            advice_items = [item.strip() for item in true_advice.split(',') if item.strip()]
            for advice in advice_items:
                if advice not in advice_stats:
                    advice_stats[advice] = {"total": 0, "correct": 0, "accuracy": 0.0}
                advice_stats[advice]["total"] += 1
                if result.get("is_accurate", False):
                    advice_stats[advice]["correct"] += 1
    for advice, stats in advice_stats.items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
    with open(args.output, 'w', encoding='utf-8') as f:
        output_data = {
            "results": results,
            "accuracy_stats": accuracy_stats,
            "advice_stats": advice_stats,
            "total_samples": start_index + len(test_data),
            "processed_samples": start_index + len(test_data),
            "valid_samples": accuracy_stats["total"],
            "failed_calls": accuracy_stats["failed_calls"],
            "evaluation_standard": "Balanced standard - 1-2 advice must all match, >2 advice at least 2 must match",
            "evaluation_method": "LLM agent" if not args.rule_based else "Rule-based",
            "enhancement": "RAG (knowledge base retrieval + patient similarity retrieval)"
        }
        if args.token_stats:
            output_data["token_stats"] = token_stats
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    return 0

if __name__ == "__main__":
    main()