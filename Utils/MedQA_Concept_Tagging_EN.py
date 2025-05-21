import json
import csv
import os
import time
from tqdm import tqdm
import re
from openai import OpenAI

API_KEY = "YOUR_API_KEY"
API_URL = "YOUR_API_URL"

input_file = "../Knowledge_Base/English_Knowledge_Base/english_processed.csv"
output_file = "../Knowledge_Base/English_Knowledge_Base/english_processed_with_icd10.csv"

BATCH_SIZE = 20
SAVE_INTERVAL = 5
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

def count_tokens(text):
    """Estimate token count: English by word count, Chinese by char count/2."""
    if not text:
        return 0
    words = len(text.split())
    return words

def call_llm_api(prompt):
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        input_tokens = count_tokens(prompt)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        result = response.choices[0].message.content
        output_tokens = count_tokens(result)
        return result, input_tokens, output_tokens
    except Exception as e:
        print(f"API call error: {e}")
        time.sleep(2)
        return None, 0, 0

def create_icd10_prompt(statement):
    """Create an English prompt for precise ICD-10 tagging, requiring 3-4 character codes (e.g., C21)."""
    prompt = f"""You are a medical classification expert responsible for assigning the most appropriate ICD-10 code to medical knowledge statements.

                Please analyze the following medical statement, identify the main medical concepts involved, and then assign the most appropriate ICD-10 code.
                Do not return only chapter code ranges (such as A00-B99), but provide specific codes to the level before the decimal point (such as A01, B20, C21, etc.).

                Clear requirements:
                1. Must return a code specific to the category level, such as C21, K52, I25, etc.
                2. Only return the code before the decimal point, no numbers after the decimal point
                3. Choose the single code that best represents the core content of this statement from all possible classifications
                4. Output only the code itself, no explanations or other content

                ICD-10 main chapter reference:
                - A00-B99: Infectious and parasitic diseases
                - C00-D48: Neoplasms
                - D50-D89: Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
                - E00-E90: Endocrine, nutritional and metabolic diseases
                - F00-F99: Mental and behavioral disorders
                - G00-G99: Diseases of the nervous system
                - H00-H59: Diseases of the eye and adnexa
                - H60-H95: Diseases of the ear and mastoid process
                - I00-I99: Diseases of the circulatory system
                - J00-J99: Diseases of the respiratory system
                - K00-K93: Diseases of the digestive system
                - L00-L99: Diseases of the skin and subcutaneous tissue
                - M00-M99: Diseases of the musculoskeletal system and connective tissue
                - N00-N99: Diseases of the genitourinary system
                - O00-O99: Pregnancy, childbirth and the puerperium
                - P00-P96: Certain conditions originating in the perinatal period
                - Q00-Q99: Congenital malformations, deformations and chromosomal abnormalities
                - R00-R99: Symptoms, signs and abnormal clinical and laboratory findings
                - S00-T98: Injury, poisoning and certain other consequences of external causes
                - V01-Y98: External causes of morbidity and mortality
                - Z00-Z99: Factors influencing health status and contact with health services
                - U00-U99: Codes for special purposes

                Medical statement: "{statement}"

                Please output only one ICD-10 code, such as "C21", without code ranges or any explanations."""
    return prompt

def normalize_icd10_code(code):
    """
    Normalize ICD-10 code, keep specific category code (e.g., C21),
    ensure correct format, fallback to range code if needed.
    """
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
    """Process a batch of statements, assign precise ICD-10 codes."""
    results = []
    for i, (statement_id, statement) in enumerate(zip(ids, statements)):
        retry_count = 0
        icd10_code = None
        while retry_count < RETRY_LIMIT and not icd10_code:
            if retry_count > 0:
                print(f"Retrying record {statement_id} (attempt {retry_count+1}/{RETRY_LIMIT})")
                time.sleep(2)
            prompt = create_icd10_prompt(statement)
            api_response, input_tokens, output_tokens = call_llm_api(prompt)
            if api_response:
                print(f"ID {statement_id} raw response: {api_response} (in: {input_tokens}, out: {output_tokens} tokens)")
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
    """Load already processed data."""
    processed_data = []
    processed_ids = set()
    try:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8', newline='') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)
                expected_headers = ['id', 'statement', 'icd10_code']
                if headers != expected_headers:
                    print(f"Warning: Output file header format incorrect. Expected: {expected_headers}, Actual: {headers}")
                for row in reader:
                    processed_data.append(row)
                    try:
                        processed_ids.add(int(row[0]))
                    except:
                        processed_ids.add(row[0])
            print(f"Loaded {len(processed_data)} previously processed records")
    except Exception as e:
        print(f"Error loading processed data: {e}")
    return processed_data, processed_ids

def process_csv_file(input_path, output_path, start_from=0):
    """Process CSV file, add precise ICD-10 labels to each row."""
    processed_data, processed_ids = load_existing_processed_data(output_path)
    try:
        with open(input_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            if headers != ['id', 'statement']:
                print(f"Warning: Input file header format incorrect. Expected: ['id', 'statement'], Actual: {headers}")
            all_rows = list(reader)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    total_rows = len(all_rows)
    print(f"Input file contains {total_rows} rows of data")
    rows_to_process = []
    for row in all_rows:
        if len(row) >= 2:
            try:
                row_id = int(row[0])
            except:
                row_id = row[0]
            if row_id not in processed_ids:
                rows_to_process.append(row)
    print(f"Need to process {len(rows_to_process)} rows of data")
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
            print(f"Processed {total_processed}/{len(rows_to_process)} records, current results saved")
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'statement', 'icd10_code'])
        writer.writerows(processed_data)
    print(f"Processing complete! Processed {total_processed} records, {failures} failures. Results saved to {output_path}")

def analyze_results(output_path):
    """Analyze results, show ICD-10 code distribution."""
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
        print("\nICD-10 Code Distribution Statistics:")
        print("-" * 50)
        print(f"{'Code':<10} {'Count':<8} {'Percentage':<10}")
        print("-" * 50)
        total = len(rows)
        for code, count in sorted_counts[:20]:
            percentage = (count / total) * 100
            print(f"{code:<10} {count:<8} {percentage:6.2f}%")
        print("-" * 50)
        print(f"Total records: {total}")
        print("\nICD-10 Chapter Distribution Statistics:")
        print("-" * 50)
        print(f"{'Chapter':<10} {'Count':<8} {'Percentage':<10}")
        print("-" * 50)
        for chapter, count in sorted_chapter_counts:
            percentage = (count / total) * 100
            print(f"{chapter:<10} {count:<8} {percentage:6.2f}%")
        print("-" * 50)
    except Exception as e:
        print(f"Error analyzing results: {e}")

def check_sample_results(output_path, num_samples=5):
    """Randomly check some results."""
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
        print("\nRandom Sample Results Check:")
        for i, row in enumerate(samples):
            if len(row) >= 3:
                print(f"\nSample {i+1}:")
                print(f"ID: {row[0]}")
                print(f"Medical statement: {row[1]}")
                print(f"ICD-10 code: {row[2]}")
                print("-" * 50)
            else:
                print(f"Row {i+1} has incorrect format")
    except Exception as e:
        print(f"Error checking sample results: {e}")

def main():
    print("Starting to process medical statements, adding precise ICD-10 codes...")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return
    process_csv_file(input_file, output_file)
    analyze_results(output_file)
    check_sample_results(output_file)

if __name__ == "__main__":
    main()