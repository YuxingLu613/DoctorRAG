#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import argparse
import logging
import tiktoken
from openai import OpenAI
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PATIENT_EMBEDDING_DIR = '../../Patient_Base'
PATIENT_EMBEDDINGS_FILE = os.path.join(PATIENT_EMBEDDING_DIR, 'embeddings.npy')
PATIENT_ID_MAP_FILE = os.path.join(PATIENT_EMBEDDING_DIR, 'id_to_index.pkl')
PATIENT_DATA_FILE = os.path.join(PATIENT_EMBEDDING_DIR, 'patient_data.pkl')
KNOWLEDGE_CSV_FILE = "../../Knowledge_Base/Chinese_Knowledge_Base/chinese_processed_with_icd10.csv"
KNOWLEDGE_EMBEDDINGS_FILE = "../../Knowledge_Base/Chinese_Knowledge_Base/chinese_embeddings.npy"
KNOWLEDGE_ORIGINAL_CSV = "../../Knowledge_Base/Chinese_Knowledge_Base/chinese_processed.csv"
KNOWLEDGE_METADATA_FILE = "../../Knowledge_Base/Chinese_Knowledge_Base/embeddings_metadata.json"
OUTPUT_DIR = '../../Outputs/Muzhi_DD/deepseek'
OUTPUT_FILENAME = 'deepseek_results_1.json'
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

API_KEY = "YOUR_API_KEY"
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
TOP_K = 5

def get_token_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            return None

token_encoder = get_token_encoder()

def count_tokens(text):
    if not text:
        return 0
    if token_encoder:
        return len(token_encoder.encode(text))
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    english_chars = re.findall(r'[a-zA-Z0-9]', text)
    other_chars = len(text) - len(chinese_chars) - len(english_chars)
    chinese_tokens = len(chinese_chars) / 1.5
    english_tokens = len(english_chars) / 4
    other_tokens = other_chars / 2
    total_tokens = chinese_tokens + english_tokens + other_tokens
    return max(1, int(total_tokens))

def call_llm_api(prompt, max_retries=3, retry_delay=2, api_key=API_KEY):
    input_tokens = count_tokens(prompt)
    output_tokens = 0
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical assistant specialized in diagnosing diseases based on patient symptoms. Always provide structured, concise responses."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                stream=False
            )
            result = response.choices[0].message.content.strip()
            output_tokens = count_tokens(result)
            return result, input_tokens, output_tokens
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None, input_tokens, 0

def create_icd10_prompt(query):
    prompt = f"""你是一位医疗分类专家，负责为患者查询分配最合适的 ICD-10 代码的第一层级（章节）。
                请分析下面的患者对话，识别其涉及的主要医学问题，然后分配最合适的 ICD-10 第一层级代码。

                请从以下 ICD-10 第一层级代码中选择一个最匹配的：
                - A00-B99: 传染病和寄生虫病
                - C00-D48: 肿瘤
                - D50-D89: 血液和造血器官疾病及某些涉及免疫机制的疾患
                - E00-E90: 内分泌、营养和代谢疾病
                - F00-F99: 精神和行为障碍
                - G00-G99: 神经系统疾病
                - H00-H59: 眼和附器疾病
                - H60-H95: 耳和乳突疾病
                - I00-I99: 循环系统疾病
                - J00-J99: 呼吸系统疾病
                - K00-K93: 消化系统疾病
                - L00-L99: 皮肤和皮下组织疾病
                - M00-M99: 肌肉骨骼系统和结缔组织疾病
                - N00-N99: 泌尿生殖系统疾病
                - O00-O99: 妊娠、分娩和产褥期
                - P00-P96: 起源于围生期的某些情况
                - Q00-Q99: 先天性畸形、变形和染色体异常
                - R00-R99: 症状、体征和临床与实验室异常所见
                - S00-T98: 损伤、中毒和外因的某些其他后果
                - V01-Y98: 疾病和死亡的外部原因
                - Z00-Z99: 影响健康状态和与保健机构接触的因素
                - U00-U99: 用于特殊目的的编码

                患者对话:
                "{query}"

                请只返回一个代码范围，例如 "A00-B99"，不要返回解释或其他内容。"""
    return prompt

def get_icd10_code(query):
    prompt = create_icd10_prompt(query)
    response, input_tokens, output_tokens = call_llm_api(prompt)
    if not response:
        return "R00-R99", input_tokens, output_tokens
    match = re.search(r'([A-Z]\d{2}-[A-Z]?\d{2})', response)
    if match:
        return match.group(1), input_tokens, output_tokens
    common_codes = [
        "A00-B99", "C00-D48", "D50-D89", "E00-E90", "F00-F99", "G00-G99",
        "H00-H59", "H60-H95", "I00-I99", "J00-J99", "K00-K93", "L00-L99",
        "M00-M99", "N00-N99", "O00-O99", "P00-P96", "Q00-Q99", "R00-R99",
        "S00-T98", "V01-Y98", "Z00-Z99", "U00-U99"
    ]
    for code in common_codes:
        if code in response:
            return code, input_tokens, output_tokens
    return "R00-R99", input_tokens, output_tokens

def load_patient_embedding_data():
    print("Loading patient embedding data...")
    id_to_index = None
    try:
        if os.path.exists(PATIENT_ID_MAP_FILE):
            with open(PATIENT_ID_MAP_FILE, 'rb') as f:
                id_to_index = pickle.load(f)
    except Exception:
        print("Failed to load patient ID map file")
    patient_data = None
    try:
        if os.path.exists(PATIENT_DATA_FILE):
            with open(PATIENT_DATA_FILE, 'rb') as f:
                patient_data = pickle.load(f)
    except Exception:
        print("Failed to load patient data file")
    embeddings = None
    try:
        if os.path.exists(PATIENT_EMBEDDINGS_FILE):
            embeddings = np.load(PATIENT_EMBEDDINGS_FILE)
    except Exception:
        print("Failed to load patient embeddings file")
    return {
        "id_to_index": id_to_index,
        "data": patient_data,
        "embeddings": embeddings
    }

def load_knowledge_data():
    knowledge_data = None
    if os.path.exists(KNOWLEDGE_CSV_FILE):
        knowledge_data = pd.read_csv(KNOWLEDGE_CSV_FILE)
    original_knowledge = None
    if os.path.exists(KNOWLEDGE_ORIGINAL_CSV):
        original_knowledge = pd.read_csv(KNOWLEDGE_ORIGINAL_CSV)
    knowledge_embeddings = None
    if os.path.exists(KNOWLEDGE_EMBEDDINGS_FILE):
        knowledge_embeddings = np.load(KNOWLEDGE_EMBEDDINGS_FILE)
    knowledge_metadata = None
    if os.path.exists(KNOWLEDGE_METADATA_FILE):
        with open(KNOWLEDGE_METADATA_FILE, 'r', encoding='utf-8') as f:
            knowledge_metadata = json.load(f)
    return {
        "data": knowledge_data,
        "original_data": original_knowledge,
        "embeddings": knowledge_embeddings,
        "metadata": knowledge_metadata
    }

def load_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)

def generate_embedding(model, text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    with torch.no_grad():
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding

def search_patient_by_embedding(model, query, patient_data, top_k=5):
    embeddings = patient_data["embeddings"]
    patient_df = patient_data["data"]
    id_to_index = patient_data["id_to_index"]
    query_embedding = generate_embedding(model, query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_similarities = similarities[top_indices]
    results = []
    for idx, similarity in zip(top_indices, top_similarities):
        patient_id = None
        for pid, index in id_to_index.items():
            if index == idx:
                patient_id = pid
                break
        if patient_id is None:
            continue
        patient_info = patient_df.loc[patient_df['id'] == int(patient_id)].to_dict('records')
        if not patient_info:
            continue
        patient_record = patient_info[0]
        results.append({
            "id": patient_id,
            "similarity": float(similarity),
            "self_repo": patient_record.get('self_repo', ''),
            "disease_tag": patient_record.get('disease_tag', '')
        })
    return results

def is_code_in_range(specific_code, range_code):
    if not specific_code or not range_code:
        return False
    range_match = re.match(r'([A-Z])(\d+)-([A-Z]?)(\d+)', range_code)
    if not range_match:
        return False
    start_letter, start_num, end_letter, end_num = range_match.groups()
    end_letter = end_letter or start_letter
    specific_match = re.match(r'([A-Z])(\d+)', specific_code)
    if not specific_match:
        return False
    specific_letter, specific_num = specific_match.groups()
    if specific_letter < start_letter or specific_letter > end_letter:
        return False
    if specific_letter == start_letter and int(specific_num) < int(start_num):
        return False
    if specific_letter == end_letter and int(specific_num) > int(end_num):
        return False
    return True

def filter_knowledge_by_icd10(icd10_code, knowledge_data):
    knowledge_csv = knowledge_data["data"]
    if 'icd10_code' not in knowledge_csv.columns:
        return knowledge_csv
    if '-' in icd10_code:
        mask = knowledge_csv['icd10_code'].apply(lambda x: is_code_in_range(x, icd10_code))
        filtered_data = knowledge_csv[mask]
        if filtered_data.empty:
            return pd.DataFrame()
        print(f"Filtered {len(filtered_data)} knowledge entries by ICD-10 range {icd10_code}")
        return filtered_data
    else:
        filtered_data = knowledge_csv[knowledge_csv['icd10_code'] == icd10_code]
        if filtered_data.empty:
            code_prefix = icd10_code[0] if icd10_code else ""
            if code_prefix:
                filtered_data = knowledge_csv[knowledge_csv['icd10_code'].str.startswith(code_prefix)]
        print(f"Filtered {len(filtered_data)} knowledge entries by ICD-10 code {icd10_code}")
        return filtered_data

def search_knowledge(model, query, icd10_code, knowledge_data, top_k=5):
    knowledge_embeddings = knowledge_data["embeddings"]
    knowledge_metadata = knowledge_data["metadata"]
    knowledge_csv = knowledge_data["data"]
    filtered_knowledge = filter_knowledge_by_icd10(icd10_code, knowledge_data)
    if filtered_knowledge.empty:
        filtered_indices = range(len(knowledge_embeddings))
    else:
        filtered_indices = []
        for idx, row in filtered_knowledge.iterrows():
            id_val = str(row['id'])
            if id_val in knowledge_metadata.get("id_to_index", {}):
                filtered_indices.append(knowledge_metadata["id_to_index"][id_val])
    query_embedding = generate_embedding(model, query)
    filtered_embeddings = knowledge_embeddings[filtered_indices]
    similarities = cosine_similarity(query_embedding.reshape(1, -1), filtered_embeddings)[0]
    top_local_indices = np.argsort(similarities)[::-1][:top_k]
    top_global_indices = [filtered_indices[i] for i in top_local_indices]
    results = []
    for i, global_idx in enumerate(top_global_indices):
        id_val = None
        for id_str, index in knowledge_metadata.get("id_to_index", {}).items():
            if index == global_idx:
                id_val = id_str
                break
        similarity = similarities[top_local_indices[i]]
        knowledge_info = None
        if id_val is not None:
            knowledge_row = knowledge_csv[knowledge_csv['id'] == int(id_val)]
            if not knowledge_row.empty:
                knowledge_info = knowledge_row.iloc[0].to_dict()
        results.append({
            "id": id_val,
            "similarity": similarity,
            "knowledge_info": knowledge_info
        })
    return results

def get_disease_list_from_patient_data(patient_data):
    try:
        df = patient_data["data"]
        diseases = df['disease_tag'].dropna().unique().tolist()
        diseases = [d for d in diseases if d.strip()]
        print(f"Extracted {len(diseases)} diseases from patient data")
        return diseases
    except Exception:
        print("Failed to get disease list, returning empty list")
        return []

def create_rag_prediction_prompt(query, icd10_code, patient_results, knowledge_results, disease_list=None):
    prompt = f"""作为医疗助手，请根据以下患者描述和参考信息，预测可能的疾病。

患者描述:
"{query}"

ICD-10分类: {icd10_code}

相关患者案例:
"""
    for i, result in enumerate(patient_results):
        prompt += f"\n案例{i+1} (相似度: {result['similarity']:.2f}):\n"
        prompt += f"- 患者描述: {result.get('self_repo', '无')}\n"
        prompt += f"- 诊断疾病: {result.get('disease_tag', '无')}\n"
    prompt += "\n\n相关医学知识:"
    for i, result in enumerate(knowledge_results):
        if result.get("knowledge_info"):
            info = result["knowledge_info"]
            prompt += f"\n知识{i+1} (相似度: {result['similarity']:.2f}):\n"
            key_fields = ['disease', 'symptom', 'treatment', 'department', 'definition']
            for field in key_fields:
                if field in info and info[field]:
                    prompt += f"- {field}: {info[field]}\n"
    if disease_list and len(disease_list) > 0:
        prompt += "\n\n你的预测必须从以下预定义的疾病列表中选择："
        max_display = 100
        if len(disease_list) > max_display:
            sampled_diseases = disease_list[:max_display]
            prompt += f"\n{', '.join(sampled_diseases)}等{len(disease_list)}种疾病"
        else:
            prompt += f"\n{', '.join(disease_list)}"
    prompt += """
            根据上述患者描述和参考信息，请提供：
            1. 最可能的疾病诊断
            2. 简要的诊断依据
            3. 可能的治疗建议

            请以如下格式回答：
            疾病: [疾病名称]
            诊断依据: [简要说明为什么做出这个诊断]
            治疗建议: [可能的治疗方案]
            """
    return prompt

def extract_prediction_result(response):
    if not response:
        return {"disease": None, "reasoning": None, "treatment": None, "raw_response": None}
    disease_match = re.search(r'疾病[:：]\s*(.*?)(?:\n|$)', response)
    disease = disease_match.group(1).strip() if disease_match else None
    reasoning_match = re.search(r'诊断依据[:：]\s*(.*?)(?:\n\w|$)', response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    treatment_match = re.search(r'治疗建议[:：]\s*(.*?)(?:\n\w|$)', response, re.DOTALL)
    treatment = treatment_match.group(1).strip() if treatment_match else None
    if not disease and not reasoning and not treatment:
        disease_alt_patterns = [
            r'(?:预测|可能的|最可能的)疾病(?:是|为)?[:：]?\s*(.*?)(?:\n|$)',
            r'诊断[:：]?\s*(.*?)(?:\n|$)',
            r'(?:患者|病人)(?:患有|得了|可能有)[:：]?\s*(.*?)(?:\n|$)'
        ]
        for pattern in disease_alt_patterns:
            match = re.search(pattern, response)
            if match:
                disease = match.group(1).strip()
                break
    return {
        "disease": disease,
        "reasoning": reasoning,
        "treatment": treatment,
        "raw_response": response
    }

def normalize_disease_name(disease):
    if not disease:
        return ""
    return disease.strip()

def predict_with_rag(query, embedding_model, patient_data, knowledge_data, disease_list=None):
    if disease_list is None or len(disease_list) == 0:
        try:
            disease_list = get_disease_list_from_patient_data(patient_data)
        except Exception:
            disease_list = []
    icd10_code, icd10_input_tokens, icd10_output_tokens = get_icd10_code(query)
    patient_results = search_patient_by_embedding(embedding_model, query, patient_data, TOP_K)
    knowledge_results = search_knowledge(embedding_model, query, icd10_code, knowledge_data, TOP_K)
    prompt = create_rag_prediction_prompt(query, icd10_code, patient_results, knowledge_results, disease_list)
    response, input_tokens, output_tokens = call_llm_api(prompt)
    prediction = extract_prediction_result(response)
    return {
        "prediction": prediction,
        "icd10_code": icd10_code,
        "patient_results": patient_results,
        "knowledge_results": knowledge_results,
        "tokens": {
            "input": icd10_input_tokens + input_tokens,
            "output": icd10_output_tokens + output_tokens,
            "total": icd10_input_tokens + input_tokens + icd10_output_tokens + output_tokens
        }
    }

def read_patient_csv(csv_file):
    if not os.path.exists(csv_file):
        return [], None
    df = pd.read_csv(csv_file)
    required_columns = ["self_repo"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        possible_columns = [col for col in df.columns if any(term in col.lower() for term in ['symptom', 'descr', 'repo', 'complaint'])]
        if possible_columns:
            query_column = possible_columns[0]
        else:
            query_column = df.columns[0]
    else:
        query_column = 'self_repo'
    queries = df[query_column].tolist()
    queries = [str(q) for q in queries if pd.notna(q) and str(q).strip()]
    return queries, df

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Disease Prediction Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def save_results(results_data, file_path=None):
    if file_path is None:
        file_path = OUTPUT_FILE_PATH
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    results_data["saved_timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    return file_path

def main():
    global TOP_K, OUTPUT_DIR, OUTPUT_FILENAME, OUTPUT_FILE_PATH
    parser = argparse.ArgumentParser(description='Medical RAG System - Dual Path Retrieval')
    parser.add_argument('--query', type=str, help='Patient symptom description')
    parser.add_argument('--input-file', type=str, help='Input file containing multiple patient queries')
    parser.add_argument('--csv-file', type=str, default='/Users/fugecheng/desktop/dxy/dxy_sampled_disease_cases.csv',
                        help='CSV file containing patient data')
    parser.add_argument('--output-file', type=str, help='Output file for results')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--api-key', type=str, default=API_KEY, help='DeepSeek API key')
    parser.add_argument('--top-k', type=int, default=TOP_K, help='Number of retrieval results')
    parser.add_argument('--process-csv', action='store_true', help='Process default CSV file')
    parser.add_argument('--continue-from', type=str, help='Continue from existing results file')
    parser.add_argument('--patient-data-dir', type=str, default=PATIENT_EMBEDDING_DIR,
                        help='Patient data directory path')
    args = parser.parse_args()

    if args.patient_data_dir:
        global PATIENT_EMBEDDING_DIR, PATIENT_EMBEDDINGS_FILE, PATIENT_ID_MAP_FILE, PATIENT_DATA_FILE
        PATIENT_EMBEDDING_DIR = args.patient_data_dir
        PATIENT_EMBEDDINGS_FILE = os.path.join(PATIENT_EMBEDDING_DIR, 'embeddings.npy')
        PATIENT_ID_MAP_FILE = os.path.join(PATIENT_EMBEDDING_DIR, 'id_to_index.pkl')
        PATIENT_DATA_FILE = os.path.join(PATIENT_EMBEDDING_DIR, 'patient_data.pkl')

    if args.top_k:
        TOP_K = args.top_k

    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.output_file:
        OUTPUT_FILE_PATH = args.output_file
        if not os.path.isabs(OUTPUT_FILE_PATH):
            OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE_PATH)
    else:
        if not args.continue_from:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            OUTPUT_FILENAME = f"rag_results_{timestamp}.json"
            OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    patient_data = load_patient_embedding_data()
    knowledge_data = load_knowledge_data()
    embedding_model = load_embedding_model()
    disease_list = get_disease_list_from_patient_data(patient_data)

    results = []
    y_true = []
    y_pred = []
    accuracy_stats = {
        "total": 0,
        "correct": 0,
        "accuracy": 0.0
    }

    start_index = 0
    if args.continue_from and os.path.exists(args.continue_from):
        with open(args.continue_from, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get("results", [])
            accuracy_stats = data.get("accuracy_stats", accuracy_stats)
            y_true = data.get("y_true", [])
            y_pred = data.get("y_pred", [])
            start_index = len(results)
        OUTPUT_FILE_PATH = args.continue_from

    if args.process_csv or (not args.query and not args.input_file):
        csv_file = args.csv_file
        queries, csv_df = read_patient_csv(csv_file)
        if not queries:
            return 1
        if start_index > 0:
            queries = queries[start_index:]
            if 'disease_tag' in csv_df.columns:
                csv_df = csv_df.iloc[start_index:].reset_index(drop=True)
        total_tokens = {"input": 0, "output": 0, "total": 0}
        for i, query in enumerate(tqdm(queries, desc="Processing queries")):
            current_index = start_index + i
            true_disease = None
            if 'disease_tag' in csv_df.columns:
                true_disease = csv_df.iloc[i]['disease_tag']
                if pd.isna(true_disease):
                    true_disease = None
                else:
                    true_disease = normalize_disease_name(str(true_disease))
            try:
                result = predict_with_rag(query, embedding_model, patient_data, knowledge_data, disease_list)
                total_tokens["input"] += result["tokens"]["input"]
                total_tokens["output"] += result["tokens"]["output"]
                total_tokens["total"] += result["tokens"]["total"]
                predicted_disease = None
                if result["prediction"]["disease"]:
                    predicted_disease = normalize_disease_name(result["prediction"]["disease"])
                is_correct = None
                if true_disease and predicted_disease:
                    is_correct = (true_disease == predicted_disease)
                    accuracy_stats["total"] += 1
                    if is_correct:
                        accuracy_stats["correct"] += 1
                    accuracy_stats["accuracy"] = accuracy_stats["correct"] / accuracy_stats["total"]
                    y_true.append(true_disease)
                    y_pred.append(predicted_disease)
                record = {
                    "index": current_index,
                    "query": query,
                    "true_disease": true_disease,
                    "prediction": result["prediction"],
                    "icd10_code": result["icd10_code"],
                    "normalized_prediction": predicted_disease,
                    "is_correct": is_correct,
                    "tokens": result["tokens"]
                }
                results.append(record)
            except Exception as e:
                results.append({
                    "index": current_index,
                    "query": query,
                    "true_disease": true_disease,
                    "error": str(e)
                })
            if (i + 1) % 10 == 0:
                output_data = {
                    "results": results,
                    "accuracy_stats": accuracy_stats,
                    "token_stats": total_tokens,
                    "total_samples": len(csv_df),
                    "processed_samples": start_index + i + 1,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "y_true": y_true,
                    "y_pred": y_pred
                }
                save_results(output_data, OUTPUT_FILE_PATH)
        if len(y_true) > 0 and len(y_pred) > 0:
            all_diseases = sorted(list(set(y_true + y_pred)))
            classification_rep = classification_report(
                y_true, y_pred, labels=all_diseases, output_dict=True
            )
            cm_output_path = os.path.join(
                OUTPUT_DIR, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plot_confusion_matrix(y_true, y_pred, all_diseases, cm_output_path)
            disease_stats = {}
            for true, pred in zip(y_true, y_pred):
                if true not in disease_stats:
                    disease_stats[true] = {"total": 0, "correct": 0, "accuracy": 0.0}
                disease_stats[true]["total"] += 1
                if true == pred:
                    disease_stats[true]["correct"] += 1
            for disease, stats in disease_stats.items():
                if stats["total"] > 0:
                    stats["accuracy"] = stats["correct"] / stats["total"]
            final_output_data = {
                "results": results,
                "accuracy_stats": accuracy_stats,
                "token_stats": total_tokens,
                "total_samples": len(csv_df),
                "processed_samples": start_index + len(queries),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "classification_report": classification_rep,
                "confusion_matrix_path": cm_output_path,
                "disease_stats": disease_stats
            }
            save_results(final_output_data, OUTPUT_FILE_PATH)
    elif args.query:
        result = predict_with_rag(args.query, embedding_model, patient_data, knowledge_data, disease_list)
        save_results(result, OUTPUT_FILE_PATH)
    elif args.input_file:
        if not os.path.exists(args.input_file):
            return 1
        queries = []
        with open(args.input_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    queries = data
                elif isinstance(data, dict) and 'queries' in data:
                    queries = data['queries']
            except json.JSONDecodeError:
                queries = [line.strip() for line in f.readlines() if line.strip()]
        if not queries:
            return 1
        results = []
        total_tokens = {"input": 0, "output": 0, "total": 0}
        for i, query in enumerate(queries):
            query_text = query
            if isinstance(query, dict):
                query_text = query.get('text', '') or query.get('query', '')
            try:
                result = predict_with_rag(query_text, embedding_model, patient_data, knowledge_data, disease_list)
                total_tokens["input"] += result["tokens"]["input"]
                total_tokens["output"] += result["tokens"]["output"]
                total_tokens["total"] += result["tokens"]["total"]
                results.append({
                    "query": query_text,
                    "prediction": result["prediction"],
                    "icd10_code": result["icd10_code"]
                })
            except Exception as e:
                results.append({
                    "query": query_text,
                    "error": str(e)
                })
            if (i + 1) % 10 == 0:
                output_data = {
                    "results": results,
                    "tokens": total_tokens,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "processed_samples": i + 1,
                    "total_samples": len(queries)
                }
                save_results(output_data, OUTPUT_FILE_PATH)
        output_data = {
            "results": results,
            "tokens": total_tokens,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "processed_samples": len(queries),
            "total_samples": len(queries)
        }
        save_results(output_data, OUTPUT_FILE_PATH)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())