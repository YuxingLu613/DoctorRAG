import json
import csv
import os
import requests
import time
from tqdm import tqdm

API_KEY = "YOUR_API_KEY"
API_URL = "YOUR_API_URL"

chinese_path = "../Knowledge_Base/MedQA/questions/Mainland/chinese_qbank.jsonl"
output_dir = "../Knowledge_Base/MedQA/processed"

os.makedirs(output_dir, exist_ok=True)

chinese_output = os.path.join(output_dir, "chinese_processed.csv")

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
        
        print(f"Abnormal API response format: {response_data}")
        return None
    except Exception as e:
        print(f"API call error: {str(e)}")
        time.sleep(2)
        return None

def process_samples(file_path, num_samples=3):
    print(f"\nProcessing {num_samples} Chinese sample questions:")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    samples = lines[:num_samples]
    
    for i, line in enumerate(samples):
        try:
            item = json.loads(line.strip())
            
            question = item.get('question', '')
            options = item.get('options', {})
            answer_key = item.get('answer', '')
            
            if isinstance(options, list):
                idx = ord(answer_key) - ord('A')
                answer_text = options[idx] if 0 <= idx < len(options) else ""
                formatted_options = {chr(ord('A') + i): opt for i, opt in enumerate(options)}
            elif isinstance(options, dict):
                answer_text = options.get(answer_key, '')
                formatted_options = options
            else:
                answer_text = ""
                formatted_options = options
            
            prompt = f"""
            我有一道医学选择题，请帮我将它转换成一个完整的陈述句，必须包含问题中的所有关键信息和正确答案，不要包含错误的选项。
            
            请确保：
            1. 陈述句必须包含题干中的所有重要信息（如患者症状、背景、诊断等）
            2. 陈述句必须自然流畅，符合医学表述习惯
            3. 陈述句必须包含正确答案，并与题干信息形成完整的医学知识点
            
            问题: {question}
            选项: {formatted_options}
            正确答案: {answer_key}. {answer_text}
            
            请直接输出转换后的完整陈述句，不要有任何额外解释。陈述句必须包含题干的全部关键信息。
            """

            processed_statement = call_llm_api(prompt)

            print(f"\nSample {i+1}:")
            print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
            print(f"Correct Answer: {answer_key}. {answer_text}")
            print(f"Processed Statement: {processed_statement}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing sample: {e}")

def process_data_file(file_path, output_path):
    """Process Chinese data file and create CSV"""
    processed_data = []
    failures = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    for idx, line in enumerate(tqdm(lines, desc="Processing Chinese data")):
        try:
            item = json.loads(line.strip())

            question = item.get('question', '')
            options = item.get('options', {})
            answer_key = item.get('answer', '')

            if isinstance(options, list):
                idx_opt = ord(answer_key) - ord('A')
                answer_text = options[idx_opt] if 0 <= idx_opt < len(options) else ""
                formatted_options = {chr(ord('A') + i): opt for i, opt in enumerate(options)}
            elif isinstance(options, dict):
                answer_text = options.get(answer_key, '')
                formatted_options = options
            else:
                answer_text = ""
                formatted_options = options

            prompt = f"""
            我有一道医学选择题，请帮我将它转换成一个完整的陈述句，必须包含问题中的所有关键信息和正确答案，不要包含错误的选项。
            
            请确保：
            1. 陈述句必须包含题干中的所有重要信息（如患者症状、背景、诊断等）
            2. 陈述句必须自然流畅，符合医学表述习惯
            3. 陈述句必须包含正确答案，并与题干信息形成完整的医学知识点
            
            问题: {question}
            选项: {formatted_options}
            正确答案: {answer_key}. {answer_text}
            
            请直接输出转换后的完整陈述句，不要有任何额外解释。陈述句必须包含题干的全部关键信息。
            """

            processed_statement = call_llm_api(prompt)

            if processed_statement:
                processed_data.append([idx + 1, processed_statement])
            else:
                failures += 1
                print(f"Line {idx+1} processing failed")
                time.sleep(3)
                processed_statement = call_llm_api(prompt)
                if processed_statement:
                    processed_data.append([idx + 1, processed_statement])
                    failures -= 1
                    print(f"Line {idx+1} retry succeeded")

            if (idx + 1) % 20 == 0 or idx == total - 1:
                with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['id', 'statement'])
                    writer.writerows(processed_data)
                print(f"Processed {idx+1}/{total} records, current results saved")

            if (idx + 1) % 10 == 0:
                time.sleep(1)

        except Exception as e:
            failures += 1
            print(f"Error processing line {idx+1}: {e}")
            time.sleep(2)

    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'statement'])
        writer.writerows(processed_data)

    print(f"Processing complete! Total: {total} records, Failed: {failures}. Results saved to {output_path}")

def check_statements(output_path, num_checks=5):
    """Check the quality of generated declarative sentences"""
    print(f"\nChecking the quality of generated declarative sentences (random {num_checks} samples):")

    try:
        with open(output_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            rows = list(reader)

        if not rows:
            print("CSV file is empty or does not exist")
            return

        import random
        samples = random.sample(rows, min(num_checks, len(rows)))

        for i, row in enumerate(samples):
            if len(row) >= 2:
                print(f"\nSample {i+1}:")
                print(f"ID: {row[0]}")
                print(f"Statement: {row[1]}")
                print("-" * 50)
            else:
                print(f"Row {i+1} format incorrect")

    except Exception as e:
        print(f"Error checking statements: {e}")

def main():
    print("Starting preprocessing of Chinese medical question data...")

    print("\n=== Sample Processing ===")
    process_samples(chinese_path, num_samples=3)

    process_all = input("\nProcess all Chinese data? (y/n): ").lower().strip() == 'y'

    if process_all:
        process_data_file(chinese_path, chinese_output)
        check_statements(chinese_output)
        print("\nChinese data processing complete!")
    else:
        print("\nOnly sample data processed.")

if __name__ == "__main__":
    main()