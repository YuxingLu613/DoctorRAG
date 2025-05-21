#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import pickle
import faiss
import json
import re
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI
import tiktoken
import torch
from sentence_transformers import SentenceTransformer

# File paths
PATIENT_FAISS_INDEX_PATH = "../../Patient_Base/DDXPlus_EN/patient_faiss.index"
KNOWLEDGE_BASE_PATH = "../../Knowledge_Base/English_Knowledge_Base/processed/english_processed_with_icd10.csv"
KNOWLEDGE_FAISS_INDEX_PATH = "../../Knowledge_Base/English_Knowledge_Base/processed/embeddings/knowledge_faiss.index"
KNOWLEDGE_METADATA_PATH = "../../Knowledge_Base/English_Knowledge_Base/processed/embeddings/knowledge_metadata.pkl"
PATIENT_METADATA_PATH = "../../Patient_Base/DDXPlus_EN/patient_metadata.pkl"
OUTPUT_DIR = "../../Outputs/DDXPlus_EN/deepseek"
TEST_DATA_PATH = "../../Datasets/DDXPlus_EN/shuffled_new_with_id.csv"
EMBEDDING_CACHE_PATH = os.path.join(OUTPUT_DIR, "embedding_cache_evidences.pkl")
RESULT_FILE = os.path.join(OUTPUT_DIR, "ddxplus_disease_predictions_evidences_only.json")
TOKEN_USAGE_LOG_PATH = os.path.join(OUTPUT_DIR, "token_usage_log.json")

MODEL_API_KEY = "YOUR_API" 
MODEL_BASE_URL = "YOUR_URL"
MODEL_NAME = "YOUR_MODEL_NAME"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_DISEASES = [
    "Pneumonia", "Influenza", "Pericarditis", "Myocarditis",
    "GERD", "Boerhaave", "Anemia", "Sarcoidosis",
    "Panic attack", "Cluster headache"
]

ICD10_CATEGORIES = {
    "A00-B99": "Certain infectious and parasitic diseases",
    "C00-D49": "Neoplasms",
    "D50-D89": "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
    "E00-E89": "Endocrine, nutritional and metabolic diseases",
    "F01-F99": "Mental, Behavioral and Neurodevelopmental disorders",
    "G00-G99": "Diseases of the nervous system",
    "H00-H59": "Diseases of the eye and adnexa",
    "H60-H95": "Diseases of the ear and mastoid process",
    "I00-I99": "Diseases of the circulatory system",
    "J00-J99": "Diseases of the respiratory system",
    "K00-K95": "Diseases of the digestive system",
    "L00-L99": "Diseases of the skin and subcutaneous tissue",
    "M00-M99": "Diseases of the musculoskeletal system and connective tissue",
    "N00-N99": "Diseases of the genitourinary system",
    "O00-O9A": "Pregnancy, childbirth and the puerperium",
    "P00-P96": "Certain conditions originating in the perinatal period",
    "Q00-Q99": "Congenital malformations, deformations and chromosomal abnormalities",
    "R00-R99": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
    "S00-T88": "Injury, poisoning and certain other consequences of external causes",
    "V00-Y99": "External causes of morbidity",
    "Z00-Z99": "Factors influencing health status and contact with health services"
}

DISEASE_TO_ICD10 = {
    "Pneumonia": "J00-J99",
    "Influenza": "J00-J99",
    "Pericarditis": "I00-I99",
    "Myocarditis": "I00-I99",
    "GERD": "K00-K95",
    "Boerhaave": "K00-K95",
    "Anemia": "D50-D89",
    "Sarcoidosis": "D50-D89",
    "Panic attack": "F01-F99",
    "Cluster headache": "G00-G99"
}

class TokenCounter:
    def __init__(self):
        self.encoders = {}
        try:
            self.encoders["default"] = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoders["default"] = tiktoken.get_encoding("p50k_base")
    
    def count_tokens(self, text: str, model: str = "default") -> int:
        if text is None:
            return 0
        encoder = self.encoders.get("default")
        if encoder:
            return len(encoder.encode(text))
        return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict[str, str]], model: str = "default") -> int:
        if not messages:
            return 0
        token_count = 0
        for message in messages:
            token_count += self.count_tokens(message.get("role", ""), model)
            token_count += self.count_tokens(message.get("content", ""), model)
            token_count += 4
        token_count += 2
        return token_count

class PatientDiseaseRAGSystem:
    def __init__(self):
        self.client = OpenAI(api_key=MODEL_API_KEY, base_url=MODEL_BASE_URL)
        self.knowledge_base = None
        self.patient_faiss_index = None
        self.knowledge_faiss_index = None
        self.patient_metadata = None
        self.knowledge_metadata = None
        self.embedding_cache = self._load_embedding_cache()
        self.token_counter = TokenCounter()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to(torch.device("cuda"))
        self.token_usage = self._load_token_usage()
        self._load_data()
    
    def _load_embedding_cache(self) -> Dict[str, np.ndarray]:
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _load_token_usage(self) -> Dict:
        if os.path.exists(TOKEN_USAGE_LOG_PATH):
            with open(TOKEN_USAGE_LOG_PATH, 'r') as f:
                return json.load(f)
        return {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "chat_completions": {
                "input_tokens": 0,
                "output_tokens": 0,
                "requests": 0
            },
            "embeddings": {
                "input_tokens": 0,
                "requests": 0
            },
            "by_function": {
                "get_icd10_category": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
                "predict_disease": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
                "get_embedding": {"input_tokens": 0, "calls": 0},
                "retrieve_knowledge": {"input_tokens": 0, "calls": 0}
            }
        }
    
    def save_token_usage(self):
        with open(TOKEN_USAGE_LOG_PATH, 'w') as f:
            json.dump(self.token_usage, f, indent=2)
    
    def save_embedding_cache(self):
        with open(EMBEDDING_CACHE_PATH, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
    
    def _load_data(self):
        self.knowledge_base = pd.read_csv(KNOWLEDGE_BASE_PATH)
        self.patient_faiss_index = faiss.read_index(PATIENT_FAISS_INDEX_PATH)
        self.knowledge_faiss_index = faiss.read_index(KNOWLEDGE_FAISS_INDEX_PATH)
        with open(PATIENT_METADATA_PATH, 'rb') as f:
            self.patient_metadata = pickle.load(f)
        with open(KNOWLEDGE_METADATA_PATH, 'rb') as f:
            self.knowledge_metadata = pickle.load(f)
    
    def get_embedding(self, text: str) -> np.ndarray:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        input_tokens = self.token_counter.count_tokens(text)
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        embedding = embedding.astype(np.float32)
        self.token_usage["total_input_tokens"] += input_tokens
        self.token_usage["embeddings"]["input_tokens"] += input_tokens
        self.token_usage["embeddings"]["requests"] += 1
        self.token_usage["by_function"]["get_embedding"]["input_tokens"] += input_tokens
        self.token_usage["by_function"]["get_embedding"]["calls"] += 1
        self.embedding_cache[text] = embedding
        if len(self.embedding_cache) % 10 == 0:
            self.save_embedding_cache()
            self.save_token_usage()
        return embedding
    
    def get_icd10_category(self, evidences: List[str], initial_evidence: str) -> str:
        all_symptoms = initial_evidence + ". " + "; ".join(evidences)
        prompt = (
            f"As a medical expert, determine the most likely ICD-10 category (first level only) for a patient with the following symptoms:\n"
            f"Symptoms: {all_symptoms}\n\n"
            f"Please select the most appropriate ICD-10 category from the following list:\n"
            f"{json.dumps(ICD10_CATEGORIES, indent=2)}\n\n"
            "Return only the category code (e.g., 'A00-B99') without any additional text or explanation.\n"
        )
        messages = [
            {"role": "system", "content": "You are a medical expert specialized in ICD-10 coding."},
            {"role": "user", "content": prompt}
        ]
        input_tokens = self.token_counter.count_messages_tokens(messages)
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
        icd_category = response.choices[0].message.content.strip()
        output_tokens = self.token_counter.count_tokens(icd_category)
        self.token_usage["total_input_tokens"] += input_tokens
        self.token_usage["total_output_tokens"] += output_tokens
        self.token_usage["chat_completions"]["input_tokens"] += input_tokens
        self.token_usage["chat_completions"]["output_tokens"] += output_tokens
        self.token_usage["chat_completions"]["requests"] += 1
        self.token_usage["by_function"]["get_icd10_category"]["input_tokens"] += input_tokens
        self.token_usage["by_function"]["get_icd10_category"]["output_tokens"] += output_tokens
        self.token_usage["by_function"]["get_icd10_category"]["calls"] += 1
        icd_pattern = r'[A-Z]\d{2}-[A-Z]\d{2}'
        if re.match(icd_pattern, icd_category):
            return icd_category
        for cat in ICD10_CATEGORIES.keys():
            if cat in icd_category:
                return cat
        return "R00-R99"
    
    def get_knowledge_by_icd10(self, icd10_category: str) -> Tuple[List[Dict], List[int]]:
        match = re.match(r'([A-Z])(\d{2})-([A-Z])(\d{2})', icd10_category)
        if not match:
            return [], []
        start_letter, start_num, end_letter, end_num = match.groups()
        start_letter_ord = ord(start_letter)
        end_letter_ord = ord(end_letter)
        icd10_column = None
        knowledge_columns = list(self.knowledge_base.columns)
        for col in ['icd10_code', 'icd10code', 'ICD10_code', 'ICD10_CODE', 'icd_10_code']:
            if col in knowledge_columns:
                icd10_column = col
                break
        if not icd10_column:
            for col in knowledge_columns:
                if 'icd' in col.lower() or 'code' in col.lower():
                    icd10_column = col
                    break
        if not icd10_column:
            return [], []
        filtered_entries = []
        filtered_indices = []
        for idx, entry in self.knowledge_base.iterrows():
            if pd.isna(entry.get(icd10_column)) or not isinstance(entry.get(icd10_column), str):
                continue
            code = entry[icd10_column]
            if len(code) >= 3 and code[0].isalpha():
                entry_letter = code[0]
                try:
                    entry_num = int(code[1:3])
                    letter_ord = ord(entry_letter)
                    if (letter_ord > start_letter_ord or 
                        (letter_ord == start_letter_ord and entry_num >= int(start_num))):
                        if (letter_ord < end_letter_ord or 
                            (letter_ord == end_letter_ord and entry_num <= int(end_num))):
                            filtered_entries.append({
                                'idx': idx,
                                'icd10code': code,
                                'statement': entry.get('statement', ''),
                                'id': entry.get('id', str(idx))
                            })
                            filtered_indices.append(idx)
                except ValueError:
                    continue
        if not filtered_entries:
            prefix = start_letter
            for idx, entry in self.knowledge_base.iterrows():
                if pd.isna(entry.get(icd10_column)) or not isinstance(entry.get(icd10_column), str):
                    continue
                code = entry[icd10_column]
                if code and len(code) >= 1 and code[0] == prefix:
                    filtered_entries.append({
                        'idx': idx,
                        'icd10code': code,
                        'statement': entry.get('statement', ''),
                        'id': entry.get('id', str(idx))
                    })
                    filtered_indices.append(idx)
        return filtered_entries, filtered_indices
    
    def retrieve_knowledge_entries(self, patient_vector: np.ndarray, icd10_filtered_entries: List[Dict],
                                  icd10_filtered_indices: List[int], top_k: int = 5) -> List[Dict]:
        if not icd10_filtered_entries:
            return []
        if len(icd10_filtered_entries) <= top_k:
            return icd10_filtered_entries
        entry_ids = [entry.get('id', '') for entry in icd10_filtered_entries]
        id_to_entry_map = {entry.get('id', ''): entry for entry in icd10_filtered_entries}
        query_vector = patient_vector.reshape(1, -1).astype('float32')
        k_search = min(100, self.knowledge_faiss_index.ntotal)
        distances, indices = self.knowledge_faiss_index.search(query_vector, k_search)
        similar_entries = []
        matches_found = 0
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.knowledge_metadata):
                entry_metadata = self.knowledge_metadata[idx]
                entry_id = entry_metadata.get('id', '')
                if entry_id in id_to_entry_map:
                    matching_entry = id_to_entry_map[entry_id]
                    similar_entries.append({
                        'id': entry_id,
                        'icd10code': matching_entry.get('icd10code', ''),
                        'statement': matching_entry.get('statement', ''),
                        'similarity_score': float(distances[0][i])
                    })
                    matches_found += 1
            if matches_found >= top_k:
                break
        if not similar_entries:
            return icd10_filtered_entries[:top_k]
        return similar_entries
    
    def get_evidences_embedding(self, evidences: List[str]) -> np.ndarray:
        evidences_text = "; ".join(evidences) if evidences else ""
        return self.get_embedding(evidences_text)
    
    def retrieve_similar_patients_by_evidences(self, evidences_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        query_vector = evidences_vector.reshape(1, -1).astype('float32')
        distances, indices = self.patient_faiss_index.search(query_vector, top_k)
        similar_patients = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.patient_metadata):
                patient = self.patient_metadata[idx]
                similar_patients.append({
                    'id': patient.get('id', ''),
                    'age': patient.get('age', ''),
                    'sex': patient.get('sex', ''),
                    'pathology': patient.get('pathology', ''),
                    'evidences': patient.get('evidences', ''),
                    'initial_evidence': patient.get('initial_evidence', ''),
                    'similarity_score': float(distances[0][i])
                })
        return similar_patients
    
    def predict_disease(self, patient_data: Dict, knowledge_entries: List[Dict], similar_patients: List[Dict]) -> Tuple[str, bool]:
        evidences = patient_data.get('EVIDENCES', [])
        knowledge_text = ""
        if knowledge_entries:
            knowledge_text = "Medical Knowledge:\n"
            for i, entry in enumerate(knowledge_entries[:5]):
                knowledge_text += f"{i+1}. ICD-10: {entry.get('icd10code', '')}\n"
                knowledge_text += f" Statement: {entry.get('statement', '')[:300]}...\n\n"
        similar_patients_text = ""
        if similar_patients:
            similar_patients_text = "Similar Patient Cases:\n"
            for i, patient in enumerate(similar_patients):
                similar_patients_text += f"{i+1}. Patient with {patient.get('pathology', 'unknown disease')}\n"
                similar_patients_text += f" Age: {patient.get('age', '')}, Sex: {patient.get('sex', '')}\n"
                similar_patients_text += f" Evidences: {patient.get('evidences', '')}\n\n"
        prompt = (
            "As a medical diagnostic expert, predict the most likely disease for this patient.\n\n"
            "Patient Information:\n"
            f"- Symptoms: {'; '.join(evidences)}\n\n"
            f"{knowledge_text}"
            f"{similar_patients_text}\n"
            "Based on the above information, determine the most likely disease from the following options only:\n"
            f"{', '.join(VALID_DISEASES)}\n\n"
            "Return only the name of the disease without any additional text.\n"
        )
        messages = [
            {"role": "system", "content": "You are a medical diagnostic expert. Answer with just the disease name."},
            {"role": "user", "content": prompt}
        ]
        input_tokens = self.token_counter.count_messages_tokens(messages)
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=50
            )
            prediction = response.choices[0].message.content.strip()
            output_tokens = self.token_counter.count_tokens(prediction)
            self.token_usage["total_input_tokens"] += input_tokens
            self.token_usage["total_output_tokens"] += output_tokens
            self.token_usage["chat_completions"]["input_tokens"] += input_tokens
            self.token_usage["chat_completions"]["output_tokens"] += output_tokens
            self.token_usage["chat_completions"]["requests"] += 1
            self.token_usage["by_function"]["predict_disease"]["input_tokens"] += input_tokens
            self.token_usage["by_function"]["predict_disease"]["output_tokens"] += output_tokens
            self.token_usage["by_function"]["predict_disease"]["calls"] += 1
            for disease in VALID_DISEASES:
                if disease.lower() in prediction.lower():
                    return disease, False
            for disease in VALID_DISEASES:
                if any(word in prediction.lower() for word in disease.lower().split()):
                    return disease, False
            return VALID_DISEASES[0], False
        except Exception:
            return "API_ERROR", True
    
    def process_test_data(self, continue_from: int = 0, max_samples: Optional[int] = None) -> Dict:
        test_data = pd.read_csv(TEST_DATA_PATH)
        if max_samples:
            test_data = test_data.iloc[:max_samples]
        results = []
        stats = {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "error_cases": 0,
            "disease_stats": {disease: {"correct": 0, "total": 0, "accuracy": 0.0} for disease in VALID_DISEASES}
        }
        if continue_from > 0 and os.path.exists(RESULT_FILE):
            with open(RESULT_FILE, 'r') as f:
                existing_data = json.load(f)
                results = existing_data.get("results", [])
                stats = existing_data.get("stats", stats)
                if "error_cases" not in stats:
                    stats["error_cases"] = 0
        for idx, row in tqdm(test_data.iloc[continue_from:].iterrows(),
                             total=len(test_data) - continue_from,
                             desc="Processing test cases"):
            patient_id = row.get('id', str(idx))
            evidences_raw = row.get('EVIDENCES', '[]')
            if isinstance(evidences_raw, str):
                if evidences_raw.startswith('[') and evidences_raw.endswith(']'):
                    try:
                        evidences = json.loads(evidences_raw)
                    except Exception:
                        evidences = evidences_raw.strip('[]').split(',')
                else:
                    evidences = [evidences_raw]
            else:
                evidences = evidences_raw
            if not isinstance(evidences, list):
                evidences = [str(evidences)]
            evidences = [e.strip(' "\'') for e in evidences if e]
            patient_data = {
                'id': patient_id,
                'AGE': row.get('AGE', ''),
                'SEX': row.get('SEX', ''),
                'EVIDENCES': evidences,
                'INITIAL_EVIDENCE': row.get('INITIAL_EVIDENCE', '')
            }
            true_pathology = row.get('PATHOLOGY', '')
            try:
                icd10_category = self.get_icd10_category(evidences, patient_data['INITIAL_EVIDENCE'])
                icd10_filtered_entries, icd10_filtered_indices = self.get_knowledge_by_icd10(icd10_category)
                evidences_vector = self.get_evidences_embedding(evidences)
                knowledge_entries = self.retrieve_knowledge_entries(
                    evidences_vector,
                    icd10_filtered_entries,
                    icd10_filtered_indices
                )
                similar_patients = self.retrieve_similar_patients_by_evidences(evidences_vector)
                predicted_disease, api_error = self.predict_disease(patient_data, knowledge_entries, similar_patients)
                result_entry = {
                    "id": patient_id,
                    "age": patient_data['AGE'],
                    "sex": patient_data['SEX'],
                    "initial_evidence": patient_data['INITIAL_EVIDENCE'],
                    "evidences": patient_data['EVIDENCES'],
                    "icd10_category": icd10_category,
                    "true_pathology": true_pathology,
                    "knowledge_count": len(knowledge_entries),
                    "similar_patients_count": len(similar_patients)
                }
                if api_error:
                    result_entry["api_error"] = True
                    result_entry["predicted_disease"] = "API_ERROR"
                    result_entry["is_correct"] = None
                    stats["error_cases"] += 1
                else:
                    is_correct = predicted_disease == true_pathology
                    stats["total"] += 1
                    if is_correct:
                        stats["correct"] += 1
                    stats["accuracy"] = stats["correct"] / stats["total"]
                    if true_pathology in VALID_DISEASES:
                        stats["disease_stats"][true_pathology]["total"] += 1
                        if is_correct:
                            stats["disease_stats"][true_pathology]["correct"] += 1
                        disease_total = stats["disease_stats"][true_pathology]["total"]
                        disease_correct = stats["disease_stats"][true_pathology]["correct"]
                        if disease_total > 0:
                            stats["disease_stats"][true_pathology]["accuracy"] = disease_correct / disease_total
                    result_entry["predicted_disease"] = predicted_disease
                    result_entry["is_correct"] = is_correct
                results.append(result_entry)
                if len(results) % 10 == 0:
                    with open(RESULT_FILE, 'w') as f:
                        json.dump({"results": results, "stats": stats}, f, indent=2)
                    self.save_token_usage()
            except Exception as e:
                results.append({
                    "id": row.get('id', str(idx)),
                    "processing_error": True,
                    "error_message": str(e),
                    "true_pathology": row.get('PATHOLOGY', '')
                })
        with open(RESULT_FILE, 'w') as f:
            json.dump({"results": results, "stats": stats}, f, indent=2)
        self.save_token_usage()
        return {"results": results, "stats": stats}
    
    def print_results_summary(self, results: Dict):
        stats = results.get("stats", {})
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        print(f"\nOverall Accuracy: {stats.get('accuracy', 0.0):.4f} ({stats.get('correct', 0)}/{stats.get('total', 0)})")
        error_cases = stats.get("error_cases", 0)
        total_processed = stats.get("total", 0) + error_cases
        print(f"API Error Cases: {error_cases}/{total_processed} ({error_cases/total_processed:.2%} of all cases)")
        print(f"Valid Cases: {stats.get('total', 0)}/{total_processed}")
        print("\nDisease-specific Accuracy:")
        disease_stats = stats.get("disease_stats", {})
        for disease, disease_stat in sorted(disease_stats.items()):
            acc = disease_stat.get("accuracy", 0.0)
            correct = disease_stat.get("correct", 0)
            total = disease_stat.get("total", 0)
            if total > 0:
                print(f" {disease}: {acc:.4f} ({correct}/{total})")
        print(f"\nTotal Input Tokens: {self.token_usage['total_input_tokens']:,}")
        print(f"Total Output Tokens: {self.token_usage['total_output_tokens']:,}")
        print(f"Total Tokens: {self.token_usage['total_input_tokens'] + self.token_usage['total_output_tokens']:,}")

def main():
    print("\nDDXPlus Disease Prediction RAG System (EVIDENCES Only)")
    import argparse
    parser = argparse.ArgumentParser(description='RAG-based Disease Prediction System')
    parser.add_argument('--continue-from', type=int, default=0, help='Continue processing from this index')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process')
    args = parser.parse_args()
    try:
        rag_system = PatientDiseaseRAGSystem()
        results = rag_system.process_test_data(
            continue_from=args.continue_from,
            max_samples=args.max_samples
        )
        rag_system.print_results_summary(results)
        print(f"\nDetailed results saved to: {RESULT_FILE}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)