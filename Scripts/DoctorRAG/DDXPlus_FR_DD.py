#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import pickle
import faiss
import json
import re
import time
import traceback
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Optional
from openai import OpenAI
import logging
import tiktoken
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_disease_prediction_french.log")
    ]
)
logger = logging.getLogger(__name__)

# Paths
PATIENT_FAISS_INDEX_PATH = "../../Patient_Base/DDXPlus_FR/patient_faiss.index"
KNOWLEDGE_BASE_PATH = "../../Knowledge_Base/French_Knowledge_Base/processed/english_processed_with_icd10.csv"
KNOWLEDGE_FAISS_INDEX_PATH = "../../Knowledge_Base/French_Knowledge_Base/processed/embeddings/knowledge_faiss.index"
KNOWLEDGE_METADATA_PATH = "../../Knowledge_Base/French_Knowledge_Base/processed/embeddings/knowledge_metadata.pkl"
PATIENT_METADATA_PATH = "../../Patient_Base/DDXPlus_FR/patient_metadata.pkl"
OUTPUT_DIR = "../../Outputs/DDXPlus_FR/deepseek"
TEST_DATA_PATH = "../../Datasets/DDXPlus_FR/shuffled_new_with_id.csv"
EMBEDDING_CACHE_PATH = os.path.join(OUTPUT_DIR, "embedding_cache_evidences.pkl")
RESULT_FILE = os.path.join(OUTPUT_DIR, "ddxplus_disease_predictions.json")
TOKEN_USAGE_LOG_PATH = os.path.join(OUTPUT_DIR, "token_usage_log.json")

MODEL_API_KEY = "YOUR_API" 
MODEL_BASE_URL = "YOUR_URL"
MODEL_NAME = "YOUR_MODEL_NAME"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_DISEASES = [
    "Pneumonie",
    "Possible influenza ou syndrome virémique typique",
    "Péricardite",
    "Myocardite",
    "RGO",
    "Syndrome de Boerhaave",
    "Anémie",
    "Sarcoïdose",
    "Attaque de panique",
    "Céphalée en grappe"
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
    "Pneumonie": "J00-J99",
    "Possible influenza ou syndrome virémique typique": "J00-J99",
    "Péricardite": "I00-I99",
    "Myocardite": "I00-I99",
    "RGO": "K00-K95",
    "Syndrome de Boerhaave": "K00-K95",
    "Anémie": "D50-D89",
    "Sarcoïdose": "D50-D89",
    "Attaque de panique": "F01-F99",
    "Céphalée en grappe": "G00-G99"
}

class TokenCounter:
    def __init__(self):
        self.encoders = {}
        try:
            self.encoders["default"] = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoders["default"] = tiktoken.get_encoding("p50k_base")

    def count_tokens(self, text: str, model: str = "default") -> int:
        if text is None:
            return 0
        encoder = self.encoders.get("default")
        if encoder:
            return len(encoder.encode(text))
        else:
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
        self.embedding_cache = self.load_embedding_cache()
        self.token_counter = TokenCounter()
        
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to(torch.device("cuda"))
        
        self.token_usage = self.load_token_usage()
        self.load_data()

    def load_token_usage(self) -> Dict:
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

    def load_embedding_cache(self) -> Dict[str, np.ndarray]:
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_embedding_cache(self):
        with open(EMBEDDING_CACHE_PATH, 'wb') as f:
            pickle.dump(self.embedding_cache, f)

    def load_data(self):
        if os.path.exists(KNOWLEDGE_BASE_PATH):
            self.knowledge_base = pd.read_csv(KNOWLEDGE_BASE_PATH)
        else:
            self.knowledge_base = pd.DataFrame(columns=['id', 'statement', 'icd10_code'])
            
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
        
        prompt = f"""En tant qu'expert médical, déterminez la catégorie CIM-10 la plus probable (premier niveau uniquement) pour un patient présentant les symptômes suivants:

                    Symptômes: {all_symptoms}

                    Veuillez sélectionner la catégorie CIM-10 la plus appropriée dans la liste suivante:
                    {json.dumps(ICD10_CATEGORIES, indent=2)}

                    Retournez uniquement le code de catégorie (par exemple, 'A00-B99') sans aucun texte ou explication supplémentaire.
                    """
        
        messages = [
            {"role": "system", "content": "Vous êtes un expert médical spécialisé dans le codage CIM-10."},
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
        
        if 'icd10_code' in knowledge_columns:
            icd10_column = 'icd10_code'
        elif 'icd10code' in knowledge_columns:
            icd10_column = 'icd10code'
        elif 'ICD10_code' in knowledge_columns:
            icd10_column = 'ICD10_code'
        elif 'ICD10_CODE' in knowledge_columns:
            icd10_column = 'ICD10_CODE'
        elif 'icd_10_code' in knowledge_columns:
            icd10_column = 'icd_10_code'
        else:
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
                            metadata_idx = idx
                            filtered_entries.append({
                                'idx': metadata_idx,
                                'icd10code': code,
                                'statement': entry.get('statement', ''),
                                'id': entry.get('id', str(idx))
                            })
                            filtered_indices.append(metadata_idx)
                except ValueError:
                    continue
                    
        if not filtered_entries:
            prefix = start_letter
            lenient_entries = []
            lenient_indices = []
            
            for idx, entry in self.knowledge_base.iterrows():
                if pd.isna(entry.get(icd10_column)) or not isinstance(entry.get(icd10_column), str):
                    continue
                    
                code = entry[icd10_column]
                if code and len(code) >= 1 and code[0] == prefix:
                    metadata_idx = idx
                    lenient_entries.append({
                        'idx': metadata_idx,
                        'icd10code': code,
                        'statement': entry.get('statement', ''),
                        'id': entry.get('id', str(idx))
                    })
                    lenient_indices.append(metadata_idx)
                    
            if lenient_entries:
                return lenient_entries, lenient_indices
                
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

    def retrieve_similar_patients_by_evidences(self, evidences_vector: np.ndarray, top_k: int = 1) -> List[Dict]:
        query_vector = evidences_vector.reshape(1, -1).astype('float32')
        
        extra_k = min(top_k * 10, self.patient_faiss_index.ntotal)
        distances, indices = self.patient_faiss_index.search(query_vector, extra_k)
        
        perfect_match_threshold = 0.9999
        
        filtered_indices = []
        filtered_distances = []
        
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            similarity = 1.0 / (1.0 + dist)
            
            if similarity < perfect_match_threshold:
                filtered_indices.append(idx)
                filtered_distances.append(dist)
                
                if len(filtered_indices) >= top_k:
                    break
                    
        if not filtered_indices:
            filtered_indices = indices[0][:top_k]
            filtered_distances = distances[0][:top_k]
            
        similar_patients = []
        for i, idx in enumerate(filtered_indices):
            if idx >= 0 and idx < len(self.patient_metadata):
                patient = self.patient_metadata[idx]
                similar_patients.append({
                    'id': patient.get('id', ''),
                    'age': patient.get('age', ''),
                    'sex': patient.get('sex', ''),
                    'pathology': patient.get('pathology', ''),
                    'evidences': patient.get('evidences', ''),
                    'initial_evidence': patient.get('initial_evidence', ''),
                    'similarity_score': 1.0 / (1.0 + float(filtered_distances[i]))
                })
                
        return similar_patients

    def predict_disease(self, patient_data: Dict, knowledge_entries: List[Dict], similar_patients: List[Dict]) -> Tuple[
        str, bool]:
        evidences = patient_data.get('EVIDENCES', [])
        
        knowledge_text = ""
        if knowledge_entries:
            knowledge_text = "Connaissances médicales:\n"
            for i, entry in enumerate(knowledge_entries[:5]):
                knowledge_text += f"{i + 1}. CIM-10: {entry.get('icd10code', '')}\n"
                knowledge_text += f"   Énoncé: {entry.get('statement', '')[:300]}...\n\n"
                
        similar_patients_text = ""
        if similar_patients:
            similar_patients_text = "Cas de patients similaires:\n"
            for i, patient in enumerate(similar_patients):
                similar_patients_text += f"{i + 1}. Patient atteint de {patient.get('pathology', 'maladie inconnue')}\n"
                similar_patients_text += f"   Âge: {patient.get('age', '')}, Sexe: {patient.get('sex', '')}\n"
                similar_patients_text += f"   Symptômes: {patient.get('evidences', '')}\n\n"
                
        prompt = f"""En tant qu'expert en diagnostic médical, prédisez la maladie la plus probable pour ce patient.

                        Informations sur le patient:
                        - Symptômes: {'; '.join(evidences)}

                        {knowledge_text}
                        {similar_patients_text}

                        Sur la base des informations ci-dessus, déterminez la maladie la plus probable parmi les options suivantes uniquement:
                        {', '.join(VALID_DISEASES)}

                        Retournez uniquement le nom de la maladie sans aucun texte supplémentaire.
                        """
        
        messages = [
            {"role": "system", "content": "Vous êtes un expert en diagnostic médical. Répondez uniquement avec le nom de la maladie."},
            {"role": "user", "content": prompt}
        ]
        
        input_tokens = self.token_counter.count_messages_tokens(messages)
        
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
            try:
                patient_id = row.get('id', str(idx))
                
                evidences_raw = row.get('EVIDENCES', '[]')
                if isinstance(evidences_raw, str):
                    if evidences_raw.startswith('[') and evidences_raw.endswith(']'):
                        try:
                            evidences = json.loads(evidences_raw)
                        except json.JSONDecodeError:
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
                        json.dump({
                            "results": results,
                            "stats": stats
                        }, f, indent=2)
                    self.save_token_usage()
                    
            except Exception as e:
                results.append({
                    "id": row.get('id', str(idx)),
                    "processing_error": True,
                    "error_message": str(e),
                    "true_pathology": row.get('PATHOLOGY', '')
                })
                
        with open(RESULT_FILE, 'w') as f:
            json.dump({
                "results": results,
                "stats": stats
            }, f, indent=2)
            
        self.save_token_usage()
        
        return {
            "results": results,
            "stats": stats
        }

def main():
    parser = argparse.ArgumentParser(description='RAG-based Disease Prediction System for DDXPlus French dataset')
    parser.add_argument('--continue-from', type=int, default=0, help='Continue processing from this index')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process')
    args = parser.parse_args()

    rag_system = PatientDiseaseRAGSystem()
    rag_system.process_test_data(
        continue_from=args.continue_from,
        max_samples=args.max_samples
    )
    
    return 0

if __name__ == "__main__":
    import argparse
    import sys
    import traceback
    
    try:
        sys.exit(main())
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
