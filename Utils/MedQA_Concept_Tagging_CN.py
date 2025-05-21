import json
import csv
import os
import requests
import time
from tqdm import tqdm
import re

API_KEY = "YOUR_API_KEY"
API_URL = "YOUR_API_URL"

input_file = "../Knowledge_Base/Chinese_Knowledge_Base/chinese_processed.csv"
output_file = "../Knowledge_Base/Chinese_Knowledge_Base/chinese_processed_with_icd10.csv"

BATCH_SIZE = 20
SAVE_INTERVAL = 10
SLEEP_INTERVAL = 1
RETRY_LIMIT = 3


ICD10_RANGES = {
    "A": "A00-B99",
    "B": "A00-B99",
    "C": "C00-D48",
    "D": "D50-D89",
    "E": "E00-E90",
    "F": "F00-F99",
    "G": "G00-G99",
    "H": "H00-H59",
    "I": "I00-I99",
    "J": "J00-J99",
    "K": "K00-K93",
    "L": "L00-L99",
    "M": "M00-M99",
    "N": "N00-N99",
    "O": "O00-O99",
    "P": "P00-P96",
    "Q": "Q00-Q99",
    "R": "R00-R99",
    "S": "S00-T98",
    "T": "S00-T98",
    "U": "U00-U99",
    "V": "V01-Y98",
    "W": "V01-Y98",
    "X": "V01-Y98",
    "Y": "V01-Y98",
    "Z": "Z00-Z99"
}

def call_glm_api(prompt):
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
            "max_tokens": 150,
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

def create_icd10_prompt(statement):
    """创建用于精确 ICD-10 标注的提示，要求返回精确到3-4位的代码（如C21）"""
    prompt = f"""你是一位医疗分类专家，负责为医学知识陈述句分配最合适的 ICD-10 代码。
                请分析下面的医学陈述句，识别其涉及的主要医学概念，然后分配最合适的 ICD-10 代码。
                不要只返回章节代码范围(如A00-B99)，而是精确到小数点前的具体代码(如A01、B20、C21等)。

                明确要求：
                1. 必须返回精确到类目级别的代码，如C21、K52、I25等
                2. 只需要返回到小数点前的代码，不需要小数点后的数字
                3. 在所有可能的分类中，选择最能代表此陈述句核心内容的一个代码
                4. 只输出代码本身，不要输出任何解释或其他内容

                ICD-10主要章节参考:
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

                医学陈述句："{statement}"

                请只输出一个ICD-10代码，例如"C21"，不要输出代码范围或任何解释。"""
    return prompt

def normalize_icd10_code(code):
    precise_pattern = r'([A-Z]\d{2}(?:\.\d+)?)'
    precise_match = re.search(precise_pattern, code)
    if precise_match:
        precise_code = precise_match.group(1).split('.')[0]
        return precise_code

    range_pattern = r'([A-Z]\d{2}-[A-Z]?\d{2})'
    range_match = re.search(range_pattern, code)
    if range_match:
        range_code = range_match.group(1)
        start_code = range_code.split('-')[0]
        return start_code

    if len(code) == 1 and code.upper() in ICD10_RANGES:
        letter = code.upper()
        range_start = ICD10_RANGES[letter].split('-')[0]
        return range_start

    partial_pattern = r'([A-Z]\d{1})'
    partial_match = re.search(partial_pattern, code)
    if partial_match:
        partial_code = partial_match.group(1)
        if len(partial_code) == 2:
            return partial_code + "0"

    any_code_pattern = r'([A-Z]\d+)'
    any_code_match = re.search(any_code_pattern, code)
    if any_code_match:
        extracted_code = any_code_match.group(1)
        if len(extracted_code) >= 3:
            return extracted_code[:3]
        else:
            return extracted_code + "0" * (3 - len(extracted_code))

    letter_pattern = r'([A-Z])'
    letter_match = re.search(letter_pattern, code)
    if letter_match:
        letter = letter_match.group(1)
        if letter in ICD10_RANGES:
            range_start = ICD10_RANGES[letter].split('-')[0]
            return range_start

    print(f"Unrecognized ICD-10 code format: {code}")
    return "R99"

def process_batch(statements, ids):
    results = []
    for i, (statement_id, statement) in enumerate(zip(ids, statements)):
        retry_count = 0
        icd10_code = None
        while retry_count < RETRY_LIMIT and not icd10_code:
            if retry_count > 0:
                print(f"Retrying record {statement_id} (attempt {retry_count+1}/{RETRY_LIMIT})")
                time.sleep(2)
            prompt = create_icd10_prompt(statement)
            api_response = call_glm_api(prompt)
            if api_response:
                print(f"ID {statement_id} raw response: {api_response}")
                normalized_code = normalize_icd10_code(api_response)
                if normalized_code:
                    icd10_code = normalized_code
                    print(f"ID {statement_id} normalized code: {icd10_code}")
            retry_count += 1
        if not icd10_code:
            icd10_code = "R99"
            print(f"ID {statement_id} classification failed, using default code: {icd10_code}")
        results.append((statement_id, statement, icd10_code))
        if (i + 1) % SLEEP_INTERVAL == 0 and i < len(statements) - 1:
            time.sleep(1)
    return results

def load_existing_processed_data(output_path):
    processed_data = []
    processed_ids = set()
    try:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8', newline='') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)
                expected_headers = ['id', 'statement', 'icd10_code']
                if headers != expected_headers:
                    print(f"Warning: Output file header mismatch. Expected: {expected_headers}, Actual: {headers}")
                for row in reader:
                    processed_data.append(row)
                    try:
                        processed_ids.add(int(row[0]))
                    except:
                        processed_ids.add(row[0])
            print(f"Loaded {len(processed_data)} processed records")
    except Exception as e:
        print(f"Error loading processed data: {e}")
    return processed_data, processed_ids

def process_csv_file(input_path, output_path, start_from=0):
    processed_data, processed_ids = load_existing_processed_data(output_path)
    try:
        with open(input_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            if headers != ['id', 'statement']:
                print(f"Warning: Input file header mismatch. Expected: ['id', 'statement'], Actual: {headers}")
            all_rows = list(reader)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    total_rows = len(all_rows)
    print(f"Input file contains {total_rows} rows")
    rows_to_process = []
    for row in all_rows:
        if len(row) >= 2:
            try:
                row_id = int(row[0])
            except:
                row_id = row[0]
            if row_id not in processed_ids:
                rows_to_process.append(row)
    print(f"{len(rows_to_process)} rows to process")
    if not rows_to_process:
        print("No new data to process")
        return
    batches = [rows_to_process[i:i+BATCH_SIZE] for i in range(0, len(rows_to_process), BATCH_SIZE)]
    failures = 0
    total_processed = 0
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        batch_ids = [row[0] for row in batch]
        batch_statements = [row[1] for row in batch]
        batch_results = process_batch(batch_statements, batch_ids)
        for result in batch_results:
            if result[2]:
                processed_data.append(result)
                try:
                    processed_ids.add(int(result[0]))
                except:
                    processed_ids.add(result[0])
                total_processed += 1
            else:
                failures += 1
        if (batch_idx + 1) % SAVE_INTERVAL == 0 or batch_idx == len(batches) - 1:
            with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['id', 'statement', 'icd10_code'])
                writer.writerows(processed_data)
            print(f"Processed {total_processed}/{len(rows_to_process)} records, results saved")
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'statement', 'icd10_code'])
        writer.writerows(processed_data)
    print(f"Processing complete! {total_processed} records processed, {failures} failed. Results saved to {output_path}")

def analyze_results(output_path):
    try:
        with open(output_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            rows = list(reader)
        if not rows:
            print("CSV file is empty or does not exist")
            return
        code_counts = {}
        chapter_counts = {}
        for row in rows:
            if len(row) >= 3:
                code = row[2]
                code_counts[code] = code_counts.get(code, 0) + 1
                if len(code) >= 1:
                    chapter = code[0]
                    if chapter in ICD10_RANGES:
                        chapter_range = ICD10_RANGES[chapter]
                        chapter_counts[chapter_range] = chapter_counts.get(chapter_range, 0) + 1
        sorted_counts = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_chapter_counts = sorted(chapter_counts.items(), key=lambda x: x[1], reverse=True)
        print("\nICD-10 Code Distribution:")
        print("-" * 50)
        print(f"{'Code':<10} {'Count':<8} {'Percent':<10}")
        print("-" * 50)
        total = len(rows)
        for code, count in sorted_counts[:20]:
            percentage = (count / total) * 100
            print(f"{code:<10} {count:<8} {percentage:6.2f}%")
        print("-" * 50)
        print(f"Total records: {total}")
        print("\nICD-10 Chapter Distribution:")
        print("-" * 50)
        print(f"{'Chapter':<10} {'Count':<8} {'Percent':<10}")
        print("-" * 50)
        for chapter, count in sorted_chapter_counts:
            percentage = (count / total) * 100
            print(f"{chapter:<10} {count:<8} {percentage:6.2f}%")
        print("-" * 50)
    except Exception as e:
        print(f"Error analyzing results: {e}")

def check_sample_results(output_path, num_samples=5):
    try:
        with open(output_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            rows = list(reader)
        if not rows:
            print("CSV file is empty or does not exist")
            return
        import random
        samples = random.sample(rows, min(num_samples, len(rows)))
        print("\nRandom sample check:")
        for i, row in enumerate(samples):
            if len(row) >= 3:
                print(f"\nSample {i+1}:")
                print(f"ID: {row[0]}")
                print(f"Statement: {row[1]}")
                print(f"ICD-10 Code: {row[2]}")
                print("-" * 50)
            else:
                print(f"Row {i+1} format incorrect")
    except Exception as e:
        print(f"Error checking sample results: {e}")

def main():
    print("Starting processing of medical statements, adding precise ICD-10 codes...")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return
    process_csv_file(input_file, output_file)
    analyze_results(output_file)
    check_sample_results(output_file)

if __name__ == "__main__":
    main()