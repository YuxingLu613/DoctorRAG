import json
import csv
import os
import requests
import time
from tqdm import tqdm

API_KEY = "YOUR_API_KEY"
API_URL = "YOUR_API_URL"

english_path = "../Knowledge_Base/MedQA/questions/US/train.jsonl"
output_dir = "../Knowledge_Base/MedQA/processed"

os.makedirs(output_dir, exist_ok=True)

english_output = os.path.join(output_dir, "english_processed.csv")

def call_llm_api(prompt):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": API_KEY
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "top_p": 0.85,
            "max_tokens": 500,
            "stream": False
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        if "choices" in response_data and len(response_data["choices"]) > 0:
            choice = response_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"].strip()
        return None
    except Exception as e:
        print(f"API call error: {e}")
        return None

def process_samples(file_path, num_samples=3):
    print(f"\nProcessing {num_samples} sample questions:")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    samples = lines[:num_samples]
    for i, line in enumerate(samples):
        try:
            item = json.loads(line.strip())
            question = item.get('question', '')
            options = item.get('options', {})
            answer_key = item.get('answer', '')
            if isinstance(options, dict):
                answer_text = options.get(answer_key, '')
                formatted_options = options
            else:
                answer_text = ""
                formatted_options = options
            prompt = f"""
            I have a medical multiple choice question. Please convert it into a complete statement that must include all key information from the question and the correct answer, without the incorrect options.

            Please ensure:
            1. The statement must include all important information from the question (such as patient symptoms, background, diagnosis, etc.)
            2. The statement must flow naturally and follow medical expression conventions
            3. The statement must include the correct answer and form a complete medical knowledge point with the question information
            
            Question: {question}
            Options: {formatted_options}
            Correct answer: {answer_key}. {answer_text}
            
            Please output only the converted complete statement without any additional explanation. The statement must include all key information from the question.
            """
            processed_statement = call_llm_api(prompt)
            print(f"\nSample {i+1}:")
            print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
            print(f"Correct answer: {answer_key}. {answer_text}")
            print(f"Processed result: {processed_statement}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing sample: {e}")

def process_data_file(file_path, output_path):
    processed_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for idx, line in enumerate(tqdm(lines, desc="Processing data")):
        try:
            item = json.loads(line.strip())
            question = item.get('question', '')
            options = item.get('options', {})
            answer_key = item.get('answer', '')
            if isinstance(options, dict):
                answer_text = options.get(answer_key, '')
                formatted_options = options
            else:
                answer_text = ""
                formatted_options = options
            prompt = f"""
            I have a medical multiple choice question. Please convert it into a complete statement that must include all key information from the question and the correct answer, without the incorrect options.

            Please ensure:
            1. The statement must include all important information from the question (such as patient symptoms, background, diagnosis, etc.)
            2. The statement must flow naturally and follow medical expression conventions
            3. The statement must include the correct answer and form a complete medical knowledge point with the question information
            
            Question: {question}
            Options: {formatted_options}
            Correct answer: {answer_key}. {answer_text}
            
            Please output only the converted complete statement without any additional explanation. The statement must include all key information from the question.
            """
            processed_statement = call_llm_api(prompt)
            if processed_statement:
                processed_data.append([idx + 1, processed_statement])
        except Exception as e:
            print(f"Error processing line {idx+1}: {e}")
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'statement'])
        writer.writerows(processed_data)
    print(f"Processing complete! Results saved to {output_path}")

def main():
    print("Starting data preprocessing...")
    print("\n=== Sample Processing ===")
    process_samples(english_path, num_samples=2)
    process_all = input("\nProcess all data? (y/n): ").lower().strip() == 'y'
    if process_all:
        process_data_file(english_path, english_output)
        print("\nAll data processing complete!")
    else:
        print("\nOnly processed sample data.")

if __name__ == "__main__":
    main()