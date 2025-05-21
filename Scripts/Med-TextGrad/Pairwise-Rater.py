import json
import itertools
import time
from openai import OpenAI

DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL_NAME = "deepseek-chat"

try:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL are correctly set.")
    exit()

def get_llm_evaluation(query_text: str, response_a_text: str, response_b_text: str, llm_client: OpenAI, model_name: str) -> str | None:
    system_prompt_content = (
        "You are a clinical expert. Your responsibility is to:\n"
        "1. Compare responses from two Retrieval-Augmented Generation (RAG) methods for the same medical query.\n"
        "2. Compare the responses across three dimensions: comprehensiveness, relevance, and safety.\n"
        "3. Provide a structured evaluation with justifications for each dimension."
    )

    user_prompt_content = (
        f"Input:\n"
        f"A medical query: \"{query_text}\"\n"
        f"Response A: \"{response_a_text}\"\n"
        f"Response B: \"{response_b_text}\"\n\n"
        f"Output format:\n"
        f"A structured evaluation in the format:\n"
        f"- Comprehensiveness: [Response A/B] - [Justification]\n"
        f"- Relevance: [Response A/B] - [Justification]\n"
        f"- Safety: [Response A/B] - [Justification]\n\n"
        f"Key Requirements:\n"
        f"- Comprehensiveness: Evaluate whether the response is thorough, covering all relevant aspects of the query.\n"
        f"- Relevance: Determine if the response directly addresses the query without extraneous information.\n"
        f"- Safety: Ensure the response is medically accurate, avoiding harmful or misleading advice and adhering to clinical best practices.\n\n"
        f"Error Handling:\n"
        f"- If a response is incomplete or unclear, note the issue in the justification and evaluate based on available content.\n"
        f"- If the query is ambiguous, base the evaluation on the most likely interpretation."
    )

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": user_prompt_content}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM API for query '{query_text[:50]}...': {e}")
        return None

def main():
    all_comparison_results = []
    base_file_path_template = "../../Outputs/Med-TextGrad/medtextgrad_result_{i}.json"

    answer_keys_and_names = {
        "ground_truth_answer": "Ground Truth",
        "ori_answer": "Original Answer",
        "t1_answer": "Refined Answer (Iteration 1)",
        "t2_answer": "Refined Answer (Iteration 2)",
        "t3_answer": "Refined Answer (Iteration 3)"
    }

    for i in range(1, 7):
        file_path = base_file_path_template.format(i=i)
        print(f"\nProcessing file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print("  File not found. Skipping.")
            continue
        except json.JSONDecodeError:
            print("  Error decoding JSON from file. Skipping.")
            continue
        except Exception as e:
            print(f"  An unexpected error occurred while reading file: {e}. Skipping.")
            continue

        query = data.get("query")
        if not query:
            print(f"  Query not found in {file_path}. Skipping.")
            continue

        try:
            answers_data = {
                "ground_truth_answer": data["reference_answer"],
                "ori_answer": data["iteration_outputs"][0]["input_answer"],
                "t1_answer": data["iteration_outputs"][0]["refined_answer"],
                "t2_answer": data["iteration_outputs"][1]["refined_answer"],
                "t3_answer": data["iteration_outputs"][2]["refined_answer"]
            }
        except (IndexError, KeyError) as e:
            print(f"  Error accessing answer data in {file_path}: {e}. Skipping.")
            continue

        current_answers_to_compare = []
        for key, name in answer_keys_and_names.items():
            if key in answers_data:
                current_answers_to_compare.append({"key": key, "name": name, "text": answers_data[key]})
            else:
                print(f"  Warning: Answer key '{key}' not found in data for file {file_path}.")

        if len(current_answers_to_compare) < 2:
            print(f"  Not enough answers to compare in {file_path}. Skipping.")
            continue

        for pair in itertools.combinations(current_answers_to_compare, 2):
            response_item_A = pair[0]
            response_item_B = pair[1]

            print(f"  Evaluating Query {i}: Comparing '{response_item_A['name']}' vs '{response_item_B['name']}'")

            evaluation_result_text = get_llm_evaluation(
                query_text=query,
                response_a_text=response_item_A['text'],
                response_b_text=response_item_B['text'],
                llm_client=client,
                model_name=DEEPSEEK_MODEL_NAME
            )

            comparison_entry = {
                "file_index": i,
                "query": query,
                "response_A_source_key": response_item_A['key'],
                "response_A_source_name": response_item_A['name'],
                "response_B_source_key": response_item_B['key'],
                "response_B_source_name": response_item_B['name'],
                "llm_evaluation": evaluation_result_text if evaluation_result_text else "Error during evaluation"
            }
            all_comparison_results.append(comparison_entry)

            if evaluation_result_text:
                print(f"    Evaluation Result:\n{evaluation_result_text}\n")
            else:
                print("    Failed to get evaluation.\n")

            time.sleep(1)

    output_file_path = "pairwise_evaluation_results.json"
    try:
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            json.dump(all_comparison_results, outfile, ensure_ascii=False, indent=4)
        print(f"\nAll evaluations successfully saved to {output_file_path}")
    except Exception as e:
        print(f"\nError saving results to JSON file: {e}")

if __name__ == "__main__":
    if DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY" or not DEEPSEEK_API_KEY:
        print("ERROR: Please set your DEEPSEEK_API_KEY in the script.")
    elif DEEPSEEK_MODEL_NAME == "deepseek-chat" and input("You are using the default 'deepseek-chat' model name. Is this the correct DeepSeek-V3 model for your task? (yes/no): ").lower() != 'yes':
        print("Please update DEEPSEEK_MODEL_NAME with the specific model you intend to use.")
    else:
        main()