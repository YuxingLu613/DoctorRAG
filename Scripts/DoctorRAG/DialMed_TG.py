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
from datetime import datetime
import argparse
import logging
import tiktoken
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TEST_DATASET_PATH = "../../Datasets/dialmed/balanced_sample.txt"
DISEASE_FILE_PATH = "../../Datasets/dialmed/disease.json"
PATIENT_ID_MAP_FILE = "../../Patient_Base/DialMed/medical_dialogues_ids.json"
PATIENT_EMBEDDINGS_FILE = "../../Patient_Base/DialMed/medical_dialogues_symptoms_faiss_index.bin"
PATIENT_CSV_FILE = "../../Patient_Base/DialMed/preprocessed_medical_dialogues.csv"
KNOWLEDGE_CSV_FILE = "../../Knowledge_Base/Chinese_Knowledge_Base/chinese_processed_with_icd10.csv"
KNOWLEDGE_EMBEDDINGS_FILE = "../../Knowledge_Base/Chinese_Knowledge_Base/chinese_embeddings.npy"
KNOWLEDGE_ORIGINAL_CSV = "../../Knowledge_Base/MedQA/processed/chinese_processed.csv"
KNOWLEDGE_METADATA_FILE = "../../Knowledge_Base/Chinese_Knowledge_Base/embeddings_metadata.json"
OUTPUT_DIR = "../../Outputs/DialMed/deepseek"
RESULT_FILE = "../../Outputs/DialMed/dialmed_rag_treatment_recommendation_results.json"
API_KEY = "YOUR_API_KEY"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

def get_token_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except:
        try:
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            return None

token_encoder = get_token_encoder()

def count_tokens(text):
    if token_encoder:
        return len(token_encoder.encode(text))
    if not text:
        return 0
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    english_chars = re.findall(r'[a-zA-Z0-9]', text)
    other_chars = len(text) - len(chinese_chars) - len(english_chars)
    chinese_tokens = len(chinese_chars) / 1.5
    english_tokens = len(english_chars) / 4
    other_tokens = other_chars / 2
    total_tokens = chinese_tokens + english_tokens + other_tokens
    return max(1, int(total_tokens))

def load_test_dataset(file_path, max_samples=None):
    data = []
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list) and len(value) > 0:
                    data = value
                    break
    elif file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    if max_samples and max_samples > 0:
        data = data[:max_samples]
    return data

def load_drug_list():
    if os.path.exists(DISEASE_FILE_PATH):
        with open(DISEASE_FILE_PATH, "r", encoding="utf-8") as f:
            drug_data = json.load(f)
            if isinstance(drug_data, dict) and "label" in drug_data:
                return drug_data["label"]
            elif isinstance(drug_data, list):
                return drug_data
    return []

def load_patient_data():
    patient_id_map = {}
    if os.path.exists(PATIENT_ID_MAP_FILE):
        with open(PATIENT_ID_MAP_FILE, "r", encoding="utf-8") as f:
            patient_id_map = json.load(f)
    patient_data = None
    if os.path.exists(PATIENT_CSV_FILE):
        patient_data = pd.read_csv(PATIENT_CSV_FILE)
    patient_index = None
    if os.path.exists(PATIENT_EMBEDDINGS_FILE):
        patient_index = faiss.read_index(PATIENT_EMBEDDINGS_FILE)
    return {
        "id_map": patient_id_map,
        "data": patient_data,
        "index": patient_index,
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
        with open(KNOWLEDGE_METADATA_FILE, "r", encoding="utf-8") as f:
            knowledge_metadata = json.load(f)
    return {
        "data": knowledge_data,
        "original_data": original_knowledge,
        "embeddings": knowledge_embeddings,
        "metadata": knowledge_metadata,
    }

def call_llm_api(prompt, max_retries=3, retry_delay=2, api_key=API_KEY):
    input_tokens = count_tokens(prompt)
    output_tokens = 0
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
                stream=False,
            )
            result = response.choices[0].message.content.strip()
            output_tokens = count_tokens(result)
            return result, input_tokens, output_tokens
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
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
    match = re.search(r"([A-Z]\d{2}-[A-Z]?\d{2})", response)
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

def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def generate_embedding(model, text):
    with torch.no_grad():
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding

def search_patient_information(model, query, patient_data, top_k=5):
    patient_index = patient_data["index"]
    patient_csv = patient_data["data"]
    patient_id_map = patient_data["id_map"]
    if patient_index is None or patient_csv is None:
        return []
    query_embedding = generate_embedding(model, query)
    query_vector = query_embedding.reshape(1, -1).astype("float32")
    distances, indices = patient_index.search(query_vector, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        patient_id = None
        if isinstance(patient_id_map, dict):
            for id_str, index in patient_id_map.items():
                if index == idx:
                    patient_id = id_str
                    break
        elif isinstance(patient_id_map, list) and idx < len(patient_id_map):
            patient_id = str(patient_id_map[idx])
        if patient_id is None:
            patient_id = str(idx)
        patient_info = None
        if "id" in patient_csv.columns:
            patient_row = patient_csv[patient_csv["id"] == int(patient_id)]
            if not patient_row.empty:
                patient_info = patient_row.iloc[0].to_dict()
        if patient_info is None and idx < len(patient_csv):
            patient_info = patient_csv.iloc[idx].to_dict()
        similarity = 1.0 / (1.0 + dist)
        results.append(
            {
                "id": patient_id,
                "similarity": similarity,
                "patient_info": patient_info,
            }
        )
    return results

def is_code_in_range(specific_code, range_code):
    if not specific_code or not range_code:
        return False
    range_match = re.match(r"([A-Z])(\d+)-([A-Z]?)(\d+)", range_code)
    if not range_match:
        return False
    start_letter, start_num, end_letter, end_num = range_match.groups()
    end_letter = end_letter or start_letter
    specific_match = re.match(r"([A-Z])(\d+)", specific_code)
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
    if knowledge_csv is None:
        return pd.DataFrame()
    if "icd10_code" not in knowledge_csv.columns:
        return knowledge_csv
    if "-" in icd10_code:
        mask = knowledge_csv["icd10_code"].apply(lambda x: is_code_in_range(x, icd10_code))
        filtered_data = knowledge_csv[mask]
        if filtered_data.empty:
            return pd.DataFrame()
        return filtered_data
    else:
        filtered_data = knowledge_csv[knowledge_csv["icd10_code"] == icd10_code]
        if filtered_data.empty:
            code_prefix = icd10_code[0] if icd10_code else ""
            if code_prefix:
                filtered_data = knowledge_csv[knowledge_csv["icd10_code"].str.startswith(code_prefix)]
        if filtered_data.empty:
            return pd.DataFrame()
        return filtered_data

def search_knowledge(model, query, icd10_code, knowledge_data, top_k=5):
    knowledge_embeddings = knowledge_data["embeddings"]
    knowledge_metadata = knowledge_data["metadata"]
    knowledge_csv = knowledge_data["data"]
    if knowledge_embeddings is None or knowledge_metadata is None or knowledge_csv is None:
        return []
    filtered_knowledge = filter_knowledge_by_icd10(icd10_code, knowledge_data)
    if filtered_knowledge is None or filtered_knowledge.empty:
        filtered_indices = range(len(knowledge_embeddings))
    else:
        filtered_indices = []
        for _, row in filtered_knowledge.iterrows():
            id_val = str(row["id"])
            if id_val in knowledge_metadata.get("id_to_index", {}):
                filtered_indices.append(knowledge_metadata["id_to_index"][id_val])
    if not filtered_indices:
        return []
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
        if knowledge_csv is not None and id_val is not None:
            knowledge_row = knowledge_csv[knowledge_csv["id"] == int(id_val)]
            if not knowledge_row.empty:
                knowledge_info = knowledge_row.iloc[0].to_dict()
        results.append(
            {
                "id": id_val,
                "similarity": similarity,
                "knowledge_info": knowledge_info,
            }
        )
    return results

def create_drug_prediction_prompt(query, disease, patient_results, knowledge_results, standard_drugs):
    prompt = f"""你是一位专业的医疗助手，负责为患者推荐最合适的药物治疗方案。
请根据以下信息，推荐三种最适合的药物，必须从提供的药物列表中选择。

患者对话:
"{query}"

已确诊疾病:
"{disease}"

相关患者案例:
"""
    for i, result in enumerate(patient_results):
        if result.get("patient_info"):
            info = result["patient_info"]
            prompt += f"\n案例{i+1} (相似度: {result['similarity']:.2f}):\n"
            for key, value in info.items():
                if key != "id":
                    prompt += f"- {key}: {value}\n"
        else:
            prompt += f"\n案例{i+1} (相似度: {result['similarity']:.2f}, ID: {result['id']})\n"
    prompt += "\n\n相关医学知识:"
    for i, result in enumerate(knowledge_results):
        if result.get("knowledge_info"):
            info = result["knowledge_info"]
            prompt += f"\n知识{i+1} (相似度: {result['similarity']:.2f}):\n"
            for key, value in info.items():
                if key != "id" and key != "icd10_code":
                    prompt += f"- {key}: {value}\n"
        else:
            prompt += f"\n知识{i+1} (相似度: {result['similarity']:.2f}, ID: {result['id']})\n"
    prompt += "\n\n你的推荐必须从以下预定义的药物列表中选择："
    if standard_drugs and len(standard_drugs) > 0:
        max_display = 150
        if len(standard_drugs) > max_display:
            sampled_drugs = standard_drugs[:max_display]
            prompt += f"\n{', '.join(sampled_drugs)}等{len(standard_drugs)}种药物"
        else:
            prompt += f"\n{', '.join(standard_drugs)}"
    prompt += """
根据上述信息和约束，请直接列出你推荐的三种最合适的药物，必须从提供的药物列表中选择。

注意：只需列出药物名称，不需要解释理由。格式如下：
药物1: 药物名称1
药物2: 药物名称2
药物3: 药物名称3
"""
    return prompt

def extract_drug_prediction(response):
    if not response:
        return None
    drug_list = []
    numbered_pattern = r"药物(\d+)[:：]\s*(.*?)(?:\n|$)"
    numbered_matches = re.findall(numbered_pattern, response)
    if numbered_matches:
        for _, drug_name in numbered_matches:
            if drug_name.strip():
                drug_list.append(drug_name.strip())
    else:
        simple_patterns = [
            r"药物[:：]\s*(.*?)(?:\n|$)",
            r"推荐的药物[:：]\s*(.*?)(?:\n|$)",
            r"(?:推荐|建议)(?:使用|服用)[:：]?\s*(.*?)(?:\n|$)",
        ]
        for pattern in simple_patterns:
            matches = re.search(pattern, response)
            if matches:
                drug_names = matches.group(1).strip()
                for drug in re.split(r"[,，、；;]", drug_names):
                    if drug.strip():
                        drug_list.append(drug.strip())
                break
    if drug_list:
        return ", ".join(drug_list[:3])
    return None

def predict_drug_with_rag(query, disease, patient_results, knowledge_results, standard_drugs):
    prompt = create_drug_prediction_prompt(
        query, disease, patient_results, knowledge_results, standard_drugs
    )
    response, input_tokens, output_tokens = call_llm_api(prompt)
    predicted_drugs = extract_drug_prediction(response)
    return predicted_drugs, input_tokens, output_tokens

def preprocess_dialog(dialog):
    if isinstance(dialog, list):
        return " ".join(dialog)
    if isinstance(dialog, str):
        return dialog
    return ""

def call_evaluation_agent(true_values, predicted_values, api_key=None):
    if not api_key:
        return False, "API key not provided", 0, 0
    true_str = ", ".join([str(item) for item in true_values]) if true_values else "None"
    pred_str = (
        predicted_values
        if isinstance(predicted_values, str)
        else ", ".join([str(item) for item in predicted_values])
        if predicted_values
        else "None"
    )
    if isinstance(predicted_values, str):
        pred_list = [item.strip() for item in re.split(r"[,，、；;]", predicted_values) if item.strip()]
    else:
        pred_list = predicted_values if isinstance(predicted_values, list) else [predicted_values]
    prompt_template = f"""
    作为医疗评估专家，请评估预测的药物与真实药物的匹配程度。
    真实药物: {true_str}
    预测药物: {pred_str}
    判断标准:
    只要预测的药物中有至少一个与真实药物接近或相同，就算准确。具体而言：
    1. 如果预测的药物包含真实的药物，算准确
    2. 如果预测的药物是真实药物的别称或通用名，算准确
    3. 如果预测的药物与真实药物在医学上高度相关（如同一类别的药物且具有相似治疗效果），算准确
    注意：现在预测结果包含三种药物推荐，只要其中任何一种符合以上标准，就认为预测准确。
    请以JSON格式输出，必须包含accurate字段(true或false)和explanation字段，格式如下:
    {{
      "accurate": true或false,
      "explanation": "详细判断依据解释"
    }}
    """
    response, input_tokens, output_tokens = call_llm_api(prompt_template, api_key=api_key)
    if response:
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result_json = json.loads(json_str)
                is_accurate = result_json.get("accurate", False)
                explanation = result_json.get("explanation", "No explanation")
                return is_accurate, explanation, input_tokens, output_tokens
        except json.JSONDecodeError:
            return False, "Failed to parse evaluation result", input_tokens, output_tokens
    return False, "Evaluation API call failed", input_tokens, output_tokens

def check_paths():
    if not os.path.exists(TEST_DATASET_PATH):
        return False
    if not os.path.exists(DISEASE_FILE_PATH):
        return False
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    return True

def run_evaluation(max_samples=None, api_key=API_KEY):
    if not check_paths():
        return None
    dataset = load_test_dataset(TEST_DATASET_PATH, max_samples)
    if not dataset:
        return None
    standard_drugs = load_drug_list()
    patient_data = load_patient_data()
    knowledge_data = load_knowledge_data()
    embedding_model = load_embedding_model()
    results = []
    matches = 0
    processed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    for i, item in enumerate(tqdm(dataset, desc="Processing test dataset")):
        true_disease = (
            item.get("disease", ["Unknown Disease"])[0]
            if isinstance(item.get("disease", []), list)
            else "Unknown Disease"
        )
        true_drug = (
            item.get("label", ["Unknown Drug"])[0]
            if isinstance(item.get("label", []), list)
            else "Unknown Drug"
        )
        dialog = item.get("dialog", [])
        original_dialog = item.get("original_dialog", dialog)
        query = preprocess_dialog(original_dialog)
        sample_input_tokens = 0
        sample_output_tokens = 0
        try:
            icd10_code, icd10_input_tokens, icd10_output_tokens = get_icd10_code(query)
            sample_input_tokens += icd10_input_tokens
            sample_output_tokens += icd10_output_tokens
            patient_results = search_patient_information(
                embedding_model, query, patient_data, TOP_K
            )
            knowledge_results = search_knowledge(
                embedding_model, query, icd10_code, knowledge_data, TOP_K
            )
            pred_drug, pred_input_tokens, pred_output_tokens = predict_drug_with_rag(
                query, true_disease, patient_results, knowledge_results, standard_drugs
            )
            sample_input_tokens += pred_input_tokens
            sample_output_tokens += pred_output_tokens
            if pred_drug:
                drug_accurate, drug_explanation, eval_input_tokens, eval_output_tokens = call_evaluation_agent(
                    [true_drug], pred_drug, api_key
                )
                sample_input_tokens += eval_input_tokens
                sample_output_tokens += eval_output_tokens
                if drug_accurate:
                    matches += 1
            else:
                drug_accurate, drug_explanation = False, "Prediction is empty"
                eval_input_tokens, eval_output_tokens = 0, 0
            processed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            pred_drug = None
            drug_accurate, drug_explanation = False, f"Error: {str(e)}"
            sample_input_tokens = 0
            sample_output_tokens = 0
        total_input_tokens += sample_input_tokens
        total_output_tokens += sample_output_tokens
        pred_drug_list = []
        if pred_drug and isinstance(pred_drug, str):
            pred_drug_list = [item.strip() for item in re.split(r"[,，、；;]", pred_drug) if item.strip()]
        detailed_patients = []
        if "patient_results" in locals() and patient_results:
            for result in patient_results:
                patient_detail = {}
                patient_detail["id"] = result.get("id", "Unknown ID")
                patient_detail["similarity"] = result.get("similarity", 0.0)
                if result.get("patient_info"):
                    patient_detail["info"] = {
                        k: v
                        for k, v in result["patient_info"].items()
                        if k not in ["id"]
                    }
                detailed_patients.append(patient_detail)
        detailed_knowledge = []
        if "knowledge_results" in locals() and knowledge_results:
            for result in knowledge_results:
                knowledge_detail = {}
                knowledge_detail["id"] = result.get("id", "Unknown ID")
                knowledge_detail["similarity"] = result.get("similarity", 0.0)
                if result.get("knowledge_info"):
                    knowledge_detail["info"] = {
                        k: v
                        for k, v in result["knowledge_info"].items()
                        if k not in ["id"]
                    }
                detailed_knowledge.append(knowledge_detail)
        results.append(
            {
                "id": i + 1,
                "true_disease": true_disease,
                "true_drug": true_drug,
                "pred_drug": pred_drug,
                "pred_drug_list": pred_drug_list,
                "dialog": dialog,
                "icd10_code": icd10_code if "icd10_code" in locals() else None,
                "patient_results_count": len(patient_results) if "patient_results" in locals() else 0,
                "knowledge_results_count": len(knowledge_results) if "knowledge_results" in locals() else 0,
                "retrieved_patients": detailed_patients,
                "retrieved_knowledge": detailed_knowledge,
                "drug_match": drug_accurate,
                "drug_explanation": drug_explanation,
                "tokens": {
                    "input": sample_input_tokens,
                    "output": sample_output_tokens,
                    "total": sample_input_tokens + sample_output_tokens,
                },
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
            current_accuracy = matches / processed if processed > 0 else 0
            with open(RESULT_FILE, "w", encoding="utf-8") as f:
                interim_result = {
                    "results": results,
                    "metrics": {
                        "processed_samples": processed,
                        "matches": matches,
                        "accuracy": current_accuracy,
                        "tokens": {
                            "input": total_input_tokens,
                            "output": total_output_tokens,
                            "total": total_input_tokens + total_output_tokens,
                        },
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                }
                json.dump(interim_result, f, ensure_ascii=False, indent=2)
    final_accuracy = matches / processed if processed > 0 else 0
    final_output = {
        "results": results,
        "metrics": {
            "processed_samples": processed,
            "matches": matches,
            "accuracy": final_accuracy,
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens,
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": {
                "top_k": TOP_K,
                "embedding_model": EMBEDDING_MODEL,
            },
        },
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    return final_output

def main():
    parser = argparse.ArgumentParser(description="RAG Drug Prediction Evaluation")
    parser.add_argument("--samples", type=int, help="Limit number of samples to process")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="LLM API key")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieval results")
    args = parser.parse_args()
    global RESULT_FILE, TOP_K
    if args.output:
        RESULT_FILE = args.output
    if args.top_k:
        TOP_K = args.top_k
    if not check_paths():
        return 1
    result = run_evaluation(
        max_samples=args.samples,
        api_key=args.api_key
    )
    if result is None:
        return 1
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())