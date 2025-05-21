#!/usr/bin/env python3

import os
import re
import time
import json
import pickle
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
import tiktoken
import torch
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NEJM_TEST_PATH = "../../Datasets/NEJM-QA/nejm_filtered/nejm_patient_cases.csv"
QUESTION_EMBEDDINGS_FILE = "../../Patient_Base/NEJM-QA/question_embeddings.json"
QUESTION_ID_MAP_FILE = "../../Patient_Base/NEJM-QA/question_id_map.json"
QUESTION_WITH_EMBEDDINGS_CSV = "../../Patient_Base/NEJM-QA/nejm_with_embeddings.csv"
KNOWLEDGE_CSV_FILE = "../../Knowledge_Base/Chinese_Knowledge_Base/chinese_processed_with_icd10.csv"
KNOWLEDGE_FAISS_INDEX = "../../Knowledge_Base/Chinese_Knowledge_Base/chinese_embeddings.npy"
KNOWLEDGE_METADATA_FILE = "../../Knowledge_Base/Chinese_Knowledge_Base/embeddings_metadata.json"
OUTPUT_DIR = "../../Outputs/NEJM-QA/deepseek"
RESULT_FILE = os.path.join(OUTPUT_DIR, "nejm_results.json")

API_KEY = "YOUR_API_KEY"
API_BASE_URL = "YOUR_BASE_URL"
MODEL_NAME = "deepseek-chat"
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
    if token_encoder:
        return len(token_encoder.encode(text))
    if not text:
        return 0
    return max(1, int(len(text) / 4))

def count_messages_tokens(messages):
    if not messages:
        return 0
    token_count = 0
    for message in messages:
        token_count += count_tokens(message.get("role", ""))
        token_count += count_tokens(message.get("content", ""))
        token_count += 4
    token_count += 2
    return token_count

def load_nejm_test_data(file_path, max_samples=None):
    df = pd.read_csv(file_path)
    if max_samples and max_samples > 0:
        df = df.head(max_samples)
    return df

def load_question_embeddings():
    question_embeddings = None
    if os.path.exists(QUESTION_EMBEDDINGS_FILE):
        with open(QUESTION_EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            question_embeddings = json.load(f)
    question_id_map = None
    if os.path.exists(QUESTION_ID_MAP_FILE):
        with open(QUESTION_ID_MAP_FILE, 'r', encoding='utf-8') as f:
            question_id_map = json.load(f)
    questions_df = None
    if os.path.exists(QUESTION_WITH_EMBEDDINGS_CSV):
        questions_df = pd.read_csv(QUESTION_WITH_EMBEDDINGS_CSV)
    return {
        "embeddings": question_embeddings,
        "id_map": question_id_map,
        "data": questions_df
    }

def load_knowledge_data():
    knowledge_data = None
    if os.path.exists(KNOWLEDGE_CSV_FILE):
        knowledge_data = pd.read_csv(KNOWLEDGE_CSV_FILE)
    knowledge_index = None
    if os.path.exists(KNOWLEDGE_FAISS_INDEX):
        knowledge_index = faiss.read_index(KNOWLEDGE_FAISS_INDEX)
    knowledge_metadata = None
    if os.path.exists(KNOWLEDGE_METADATA_FILE):
        with open(KNOWLEDGE_METADATA_FILE, 'rb') as f:
            knowledge_metadata = pickle.load(f)
        if isinstance(knowledge_metadata, list):
            id_to_index = {}
            index_to_id = {}
            for idx, item in enumerate(knowledge_metadata):
                if isinstance(item, dict):
                    if 'id' in item and 'index' in item:
                        id_to_index[str(item['id'])] = item['index']
                        index_to_id[str(item['index'])] = str(item['id'])
                    elif 'id' in item and 'embedding_index' in item:
                        id_to_index[str(item['id'])] = item['embedding_index']
                        index_to_id[str(item['embedding_index'])] = str(item['id'])
                    elif 'id' in item:
                        id_to_index[str(item['id'])] = idx
                        index_to_id[str(idx)] = str(item['id'])
                else:
                    id_to_index[str(item)] = idx
                    index_to_id[str(idx)] = str(item)
            knowledge_metadata = {
                "id_to_index": id_to_index,
                "index_to_id": index_to_id
            }
    if knowledge_metadata is None and knowledge_data is not None:
        id_to_index = {}
        index_to_id = {}
        for idx, row in knowledge_data.iterrows():
            doc_id = str(row['id'])
            id_to_index[doc_id] = idx
            index_to_id[str(idx)] = doc_id
        knowledge_metadata = {
            "id_to_index": id_to_index,
            "index_to_id": index_to_id
        }
    return {
        "data": knowledge_data,
        "index": knowledge_index,
        "metadata": knowledge_metadata
    }

def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def create_icd10_prompt(query):
    prompt = (
        "You are a medical classification expert responsible for assigning the most appropriate ICD-10 code chapter (first level) to medical questions.\n"
        "Please analyze the medical question below, identify the main medical issue, and assign the most appropriate ICD-10 first-level code.\n\n"
        "Please select one most matching code from the following ICD-10 first-level codes:\n"
        "- A00-B99: Certain infectious and parasitic diseases\n"
        "- C00-D48: Neoplasms\n"
        "- D50-D89: Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism\n"
        "- E00-E90: Endocrine, nutritional and metabolic diseases\n"
        "- F00-F99: Mental and behavioural disorders\n"
        "- G00-G99: Diseases of the nervous system\n"
        "- H00-H59: Diseases of the eye and adnexa\n"
        "- H60-H95: Diseases of the ear and mastoid process\n"
        "- I00-I99: Diseases of the circulatory system\n"
        "- J00-J99: Diseases of the respiratory system\n"
        "- K00-K93: Diseases of the digestive system\n"
        "- L00-L99: Diseases of the skin and subcutaneous tissue\n"
        "- M00-M99: Diseases of the musculoskeletal system and connective tissue\n"
        "- N00-N99: Diseases of the genitourinary system\n"
        "- O00-O99: Pregnancy, childbirth and the puerperium\n"
        "- P00-P96: Certain conditions originating in the perinatal period\n"
        "- Q00-Q99: Congenital malformations, deformations and chromosomal abnormalities\n"
        "- R00-R99: Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified\n"
        "- S00-T98: Injury, poisoning and certain other consequences of external causes\n"
        "- V01-Y98: External causes of morbidity and mortality\n"
        "- Z00-Z99: Factors influencing health status and contact with health services\n"
        "- U00-U99: Codes for special purposes\n\n"
        f"Medical question:\n\"{query}\"\n\n"
        "Please return only a code range, e.g. \"A00-B99\", without any explanation or other content."
    )
    return prompt

def call_deepseek_api(prompt, max_retries=3, retry_delay=2, api_key=API_KEY):
    input_tokens = count_tokens(prompt)
    output_tokens = 0
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant"},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
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
    return None, input_tokens, 0

def get_icd10_code(query):
    prompt = create_icd10_prompt(query)
    response, input_tokens, output_tokens = call_deepseek_api(prompt)
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

def extract_main_question(text):
    if not isinstance(text, str):
        return ''
    option_pattern = r'\n[A-D]\.'
    match = re.search(option_pattern, text)
    if match:
        return text[:match.start()].strip()
    return text.strip()

def generate_embedding(model, text):
    with torch.no_grad():
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding

def is_code_in_range(specific_code, range_code):
    if not specific_code or not range_code or not isinstance(specific_code, str) or not isinstance(range_code, str):
        return False
    range_match = re.match(r'([A-Z])(\d+)-([A-Z]?)(\d+)', range_code)
    if not range_match:
        return False
    specific_match = re.match(r'([A-Z])(\d+)', specific_code)
    if not specific_match:
        return False
    start_letter, start_num, end_letter, end_num = range_match.groups()
    end_letter = end_letter or start_letter
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
    if 'icd10_code' not in knowledge_csv.columns:
        return knowledge_csv
    if '-' in icd10_code:
        mask = knowledge_csv['icd10_code'].apply(
            lambda x: is_code_in_range(x, icd10_code) if pd.notna(x) else False)
        filtered_data = knowledge_csv[mask]
        if filtered_data.empty:
            return pd.DataFrame()
        return filtered_data
    filtered_data = knowledge_csv[knowledge_csv['icd10_code'] == icd10_code]
    if filtered_data.empty:
        code_prefix = icd10_code[0] if icd10_code else ""
        if code_prefix:
            filtered_data = knowledge_csv[
                knowledge_csv['icd10_code'].str.startswith(code_prefix) & knowledge_csv['icd10_code'].notna()]
    if filtered_data.empty:
        return pd.DataFrame()
    return filtered_data

def search_knowledge_faiss(model, query, icd10_code, knowledge_data, top_k=5):
    knowledge_index = knowledge_data["index"]
    knowledge_metadata = knowledge_data["metadata"]
    knowledge_csv = knowledge_data["data"]
    if knowledge_index is None or knowledge_metadata is None or knowledge_csv is None:
        return []
    query_embedding = generate_embedding(model, query)
    query_vector = np.array([query_embedding], dtype=np.float32)
    distances, indices = knowledge_index.search(query_vector, top_k * 2)
    filtered_results = []
    for i, idx in enumerate(indices[0]):
        entry_id = None
        index_to_id = knowledge_metadata.get("index_to_id", {})
        if str(idx) in index_to_id:
            entry_id = index_to_id[str(idx)]
        else:
            continue
        knowledge_info = None
        if knowledge_csv is not None and entry_id is not None:
            try:
                id_to_lookup = int(entry_id) if entry_id.isdigit() else entry_id
                knowledge_row = knowledge_csv[knowledge_csv['id'] == id_to_lookup]
                if not knowledge_row.empty:
                    knowledge_info = knowledge_row.iloc[0].to_dict()
            except Exception:
                knowledge_row = knowledge_csv[knowledge_csv['id'] == entry_id]
                if not knowledge_row.empty:
                    knowledge_info = knowledge_row.iloc[0].to_dict()
        if knowledge_info is None:
            continue
        if icd10_code and knowledge_info.get('icd10_code'):
            entry_icd10 = knowledge_info.get('icd10_code')
            if '-' in icd10_code:
                if not is_code_in_range(entry_icd10, icd10_code):
                    continue
            else:
                if entry_icd10 != icd10_code and not entry_icd10.startswith(icd10_code[0]):
                    continue
        similarity = 1.0 / (1.0 + distances[0][i])
        filtered_results.append({
            "id": entry_id,
            "similarity": similarity,
            "knowledge_info": knowledge_info
        })
        if len(filtered_results) >= top_k:
            break
    if not filtered_results and not icd10_code:
        for i, idx in enumerate(indices[0][:top_k]):
            entry_id = None
            index_to_id = knowledge_metadata.get("index_to_id", {})
            if str(idx) in index_to_id:
                entry_id = index_to_id[str(idx)]
            similarity = 1.0 / (1.0 + distances[0][i])
            knowledge_info = None
            if knowledge_csv is not None and entry_id is not None:
                try:
                    id_to_lookup = int(entry_id) if entry_id.isdigit() else entry_id
                    knowledge_row = knowledge_csv[knowledge_csv['id'] == id_to_lookup]
                    if not knowledge_row.empty:
                        knowledge_info = knowledge_row.iloc[0].to_dict()
                except Exception:
                    pass
            filtered_results.append({
                "id": entry_id,
                "similarity": similarity,
                "knowledge_info": knowledge_info
            })
    return filtered_results

def prepare_question_index(question_data):
    question_embeddings = question_data["embeddings"]
    if not question_embeddings:
        return None
    vectors = []
    ids = []
    for q_id, q_data in question_embeddings.items():
        if "embedding" in q_data:
            vectors.append(q_data["embedding"])
            ids.append(int(q_id))
    if not vectors:
        return None
    vectors_np = np.array(vectors, dtype=np.float32)
    d = vectors_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors_np)
    id_map = {i: q_id for i, q_id in enumerate(ids)}
    return {
        "index": index,
        "id_map": id_map
    }

def search_similar_questions(model, query, question_index, question_data, top_k=5):
    if not question_index or not question_index["index"]:
        return []
    main_query = extract_main_question(query)
    query_embedding = generate_embedding(model, main_query)
    query_vector = np.array([query_embedding], dtype=np.float32)
    distances, indices = question_index["index"].search(query_vector, top_k)
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        question_id = question_index["id_map"].get(idx)
        if question_id is None:
            continue
        question_info = None
        if question_data["embeddings"] and str(question_id) in question_data["embeddings"]:
            question_info = question_data["embeddings"][str(question_id)]
        similarity = 1.0 / (1.0 + dist)
        results.append({
            "id": question_id,
            "similarity": similarity,
            "question_info": question_info
        })
    return results

def call_deepseek_api_for_qa(prompt, max_retries=3, retry_delay=2, api_key=API_KEY):
    system_message = (
        "You are a medical expert answering multiple-choice medical exam questions.\n"
        "These are single-choice questions where only one option (A, B, C, D) is correct.\n"
        "Your task is to select the ONE correct answer choice based on the medical scenario presented.\n"
        "IMPORTANT: Respond ONLY with the letter of your answer (A, B, C, or D).\n"
        "For example: if B is correct, respond with \"B\" (not \"Option B\" or \"The answer is B\")."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=50,
                stream=False
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return ""

def extract_answer_letter(answer_text):
    if not answer_text:
        return ""
    normalized_text = re.sub(r'[^A-Za-z]', '', answer_text).upper()
    valid_letters = [c for c in normalized_text if c in 'ABCD']
    if valid_letters:
        return valid_letters[0]
    return ""

def compare_answers(model_answer, correct_answer):
    model_letter = extract_answer_letter(model_answer)
    correct_letter = extract_answer_letter(correct_answer)
    return model_letter == correct_letter

def create_rag_prompt(question, similar_questions, knowledge_results):
    prompt = f"Please answer the following medical multiple-choice question:\n\n{question}\n\nHere is some relevant information that may help with answering:\n\n"
    if similar_questions and len(similar_questions) > 0:
        prompt += "## Similar Questions for Reference\n\n"
        for i, result in enumerate(similar_questions):
            info = result.get("question_info", {})
            if info:
                original_question = info.get("question", "")
                if original_question:
                    prompt += f"Similar Question {i + 1} (Similarity: {result['similarity']:.2f}):\n"
                    prompt += f"{original_question}\n"
                    if "answer" in info:
                        prompt += f"Answer: {info.get('answer', '')}\n\n"
    if knowledge_results and len(knowledge_results) > 0:
        prompt += "## Relevant Medical Knowledge\n\n"
        for i, result in enumerate(knowledge_results):
            info = result.get("knowledge_info", {})
            if info:
                prompt += f"Knowledge Point {i + 1} (Similarity: {result['similarity']:.2f}):\n"
                useful_fields = ['title', 'content', 'description', 'symptoms', 'treatment', 'diagnosis']
                for field in useful_fields:
                    if field in info and info[field]:
                        prompt += f"- {field}: {info[field]}\n"
                prompt += "\n"
    prompt += "Please carefully analyze the above relevant information and select the correct answer to the question. Only provide the option letter (A, B, C, or D)."
    return prompt

def main():
    global RESULT_FILE, TOP_K, API_KEY, API_BASE_URL, MODEL_NAME
    parser = argparse.ArgumentParser(description='NEJM Medical Question RAG System')
    parser.add_argument('--samples', type=int, help='Limit the number of samples to process')
    parser.add_argument('--api-key', type=str, default=API_KEY, help='DeepSeek API key')
    parser.add_argument('--api-base', type=str, default=API_BASE_URL, help='DeepSeek API base URL')
    parser.add_argument('--model', type=str, default=MODEL_NAME, help='DeepSeek model name')
    parser.add_argument('--output', type=str, help='Result output file path')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--baseline', action='store_true', help='Run baseline test (without RAG)')
    args = parser.parse_args()
    if args.output:
        RESULT_FILE = args.output
    if args.top_k:
        TOP_K = args.top_k
    if args.api_key:
        API_KEY = args.api_key
    if args.api_base:
        API_BASE_URL = args.api_base
    if args.model:
        MODEL_NAME = args.model
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    nejm_data = load_nejm_test_data(NEJM_TEST_PATH, args.samples)
    question_data = load_question_embeddings()
    knowledge_data = load_knowledge_data()
    embedding_model = load_embedding_model()
    question_index = prepare_question_index(question_data)
    results = []
    token_usage = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "chat_completions": {
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0
        },
        "by_function": {
            "answer_question": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            "get_icd10": {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        }
    }
    for i, row in tqdm(nejm_data.iterrows(), total=len(nejm_data), desc="Processing questions"):
        question = row['question']
        correct_answer = row['answer'].strip().upper()
        if args.baseline:
            api_response = call_deepseek_api_for_qa(question)
            model_answer = extract_answer_letter(api_response)
            is_correct = compare_answers(model_answer, correct_answer)
            token_usage["chat_completions"]["requests"] += 1
            token_usage["chat_completions"]["input_tokens"] += count_tokens(question)
            token_usage["chat_completions"]["output_tokens"] += count_tokens(api_response)
            token_usage["by_function"]["answer_question"]["calls"] += 1
            token_usage["by_function"]["answer_question"]["input_tokens"] += count_tokens(question)
            token_usage["by_function"]["answer_question"]["output_tokens"] += count_tokens(api_response)
            results.append({
                "id": i,
                "question": question,
                "correct_answer": correct_answer,
                "model_answer": model_answer,
                "full_response": api_response,
                "is_correct": is_correct,
                "mode": "baseline"
            })
        else:
            try:
                icd10_code, input_tokens, output_tokens = get_icd10_code(question)
                token_usage["total_input_tokens"] += input_tokens
                token_usage["total_output_tokens"] += output_tokens
                token_usage["by_function"]["get_icd10"]["calls"] += 1
                token_usage["by_function"]["get_icd10"]["input_tokens"] += input_tokens
                token_usage["by_function"]["get_icd10"]["output_tokens"] += output_tokens
                similar_questions = []
                if question_index:
                    similar_questions = search_similar_questions(
                        embedding_model,
                        question,
                        question_index,
                        question_data,
                        TOP_K
                    )
                knowledge_results = []
                if knowledge_data["index"] is not None:
                    knowledge_results = search_knowledge_faiss(
                        embedding_model,
                        question,
                        icd10_code,
                        knowledge_data,
                        TOP_K
                    )
                rag_prompt = create_rag_prompt(question, similar_questions, knowledge_results)
                api_response = call_deepseek_api_for_qa(rag_prompt)
                model_answer = extract_answer_letter(api_response)
                is_correct = compare_answers(model_answer, correct_answer)
                qa_input_tokens = count_tokens(rag_prompt)
                qa_output_tokens = count_tokens(api_response)
                token_usage["total_input_tokens"] += qa_input_tokens
                token_usage["total_output_tokens"] += qa_output_tokens
                token_usage["chat_completions"]["input_tokens"] += qa_input_tokens
                token_usage["chat_completions"]["output_tokens"] += qa_output_tokens
                token_usage["chat_completions"]["requests"] += 1
                token_usage["by_function"]["answer_question"]["input_tokens"] += qa_input_tokens
                token_usage["by_function"]["answer_question"]["output_tokens"] += qa_output_tokens
                token_usage["by_function"]["answer_question"]["calls"] += 1
                results.append({
                    "id": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "model_answer": model_answer,
                    "full_response": api_response,
                    "is_correct": is_correct,
                    "icd10_code": icd10_code,
                    "similar_questions_count": len(similar_questions),
                    "knowledge_results_count": len(knowledge_results),
                    "mode": "rag"
                })
            except Exception as e:
                results.append({
                    "id": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "error": str(e),
                    "is_correct": False,
                    "mode": "error"
                })
        if (i + 1) % 10 == 0 or (i + 1) == len(nejm_data):
            correct_count = sum(1 for r in results if r.get("is_correct", False))
            total_processed = len(results)
            current_accuracy = correct_count / total_processed if total_processed > 0 else 0
            temp_results = {
                "results": results,
                "stats": {
                    "total": total_processed,
                    "correct": correct_count,
                    "incorrect": sum(1 for r in results if "is_correct" in r and not r["is_correct"]),
                    "error": sum(1 for r in results if "error" in r),
                    "accuracy": current_accuracy
                },
                "token_usage": token_usage
            }
            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(temp_results, f, ensure_ascii=False, indent=2)
    stats = {
        "total": len(results),
        "correct": sum(1 for r in results if r.get("is_correct", False)),
        "incorrect": sum(1 for r in results if "is_correct" in r and not r["is_correct"]),
        "error": sum(1 for r in results if "error" in r)
    }
    stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    mode_stats = {}
    for result in results:
        mode = result.get("mode", "unknown")
        if mode not in mode_stats:
            mode_stats[mode] = {"total": 0, "correct": 0, "accuracy": 0.0}
        mode_stats[mode]["total"] += 1
        if result.get("is_correct", False):
            mode_stats[mode]["correct"] += 1
    for mode, m_stats in mode_stats.items():
        if m_stats["total"] > 0:
            m_stats["accuracy"] = m_stats["correct"] / m_stats["total"]
    final_results = {
        "results": results,
        "stats": stats,
        "mode_stats": mode_stats,
        "token_usage": token_usage,
        "params": {
            "top_k": TOP_K,
            "embedding_model": EMBEDDING_MODEL,
            "baseline": args.baseline,
            "llm_model": MODEL_NAME,
            "api_base": API_BASE_URL
        },
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())