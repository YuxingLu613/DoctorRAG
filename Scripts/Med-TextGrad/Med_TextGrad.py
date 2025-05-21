import json
import os
from openai import OpenAI, OpenAIError
import time
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score_calculate
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sacremoses")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

DEFAULT_ITERATIONS = 3
LLM_TIMEOUT_SECONDS = 90

class MedTextGradOptimizer:
    def __init__(self, api_key: str, base_url: str, model_name: str, num_iterations=DEFAULT_ITERATIONS, verbose=True):
        if not api_key:
            raise ValueError("API key is required to use the DeepSeek LLM.")
        if not base_url:
            raise ValueError("Base URL is required for the DeepSeek API.")
        if not model_name:
            raise ValueError("Model name is required for the DeepSeek API.")
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.history = []
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self._log(f"MedTextGradOptimizer initialized to use model '{self.model_name}' at '{base_url}'.")

    def _log(self, message):
        if self.verbose:
            print(message)
        self.history.append(message)

    def _call_deepseek_llm(self, task_description: str, prompt_content: str, max_tokens=8192, temperature=0.5) -> str:
        self._log(f"\n[[[ Calling DeepSeek LLM for: {task_description} ]]]\nPrompt (first 500 chars):\n{prompt_content[:500]}...\n")
        try:
            system_message_content = (
                "You are a sophisticated AI assistant specializing in refining medical text based on critiques and context, ensuring accuracy and relevance while preserving core meaning. Output only the requested text without preamble or explanation unless otherwise specified."
            )
            if "Answer Generation" in task_description or "Prompt Update" in task_description:
                system_message_content = (
                    "You are an AI assistant that generates and refines prompts or answers for a medical question-answering system. Focus on clarity, accuracy, and adherence to instructions. Output only the requested text without preamble or explanation unless otherwise specified."
                )
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message_content},
                    {"role": "user", "content": prompt_content}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=LLM_TIMEOUT_SECONDS
            )
            response_content = completion.choices[0].message.content
            self._log(f"LLM Response (first 500 chars):\n{response_content[:500]}...\n")
            return response_content if response_content else ""
        except OpenAIError as e:
            self._log(f"Error calling DeepSeek LLM for '{task_description}': {e}")
            return f"[LLM Call Failed for {task_description}: {str(e)}]"
        except Exception as e:
            self._log(f"An unexpected error occurred during LLM call for '{task_description}': {e}")
            return f"[Unexpected LLM Call Error for {task_description}: {str(e)}]"

    def _format_patient_case(self, case_data: dict) -> str:
        p_j = case_data.get('question', 'N/A')
        s_j_diagnosis = case_data.get('disease', 'N/A')
        s_j_advice = case_data.get('advice', 'N/A')
        a_j_id = case_data.get('id', 'N/A')
        case_specific_context = case_data.get('similar_patients', 'N/A')
        case_answer = case_data.get('answer', 'N/A')
        return (
            f"Retrieved Patient Case (ID: {a_j_id}):\n"
            f"  Patient's Query in Case: {p_j}\n"
            f"  Case-Specific Information/Context: {case_specific_context}\n"
            f"  Diagnosis in Case: {s_j_diagnosis}\n"
            f"  Advice/Treatment in Case: {s_j_advice}\n"
            f"  Recorded Answer in Case: {case_answer}\n"
        )

    def _aggregate_context(self, general_knowledge_text: str, retrieved_patient_cases: list) -> str:
        formatted_cases_text = "\n\n".join(
            [self._format_patient_case(case) for case in retrieved_patient_cases]
        )
        return (
            f"--- General Medical Knowledge (Expertise) ---\n"
            f"{general_knowledge_text}\n\n"
            f"--- Retrieved Patient Cases (Experience) ---\n"
            f"{formatted_cases_text}"
        )

    def generator_agent(self, prompt_p: str, answer_a: str) -> str:
        full_prompt = (
            f"{prompt_p}\n\n"
            f"--- Current Answer (A) to Refine ---\n"
            f"{answer_a}"
        )
        return self._call_deepseek_llm("Answer Generation/Refinement (Generator Agent)", full_prompt, max_tokens=8192, temperature=0.5)

    def knowledge_criterion_agent(self, answer_a: str, context_c: str) -> str:
        prompt = (
            f"Given the following medical context (C) (excerpt):\n{context_c[:4000]}\n\n"
            f"And the following generated answer (A):\n{answer_a}\n\n"
            f"Critique the answer (A) focusing ONLY on its factual alignment, consistency, and completeness with respect to the provided medical context (C). "
            f"Provide specific, concise points of criticism or confirmation. These are KNOWLEDGE-FOCUSED critiques. "
            f"Identify areas where the answer might misrepresent or omit crucial information from the context. "
            f"Output ONLY the critique."
        )
        return self._call_deepseek_llm("Knowledge Criterion Assessment", prompt, max_tokens=8192)

    def patient_criterion_agent(self, answer_a: str, query_q: str) -> str:
        prompt = (
            f"Given the original patient query (q):\n{query_q}\n\n"
            f"And the following generated answer (A):\n{answer_a}\n\n"
            f"Critique the answer (A) focusing ONLY on its relevance, appropriateness, and specificity to the patient's query (q). "
            f"Provide specific, concise points of criticism or confirmation. These are PATIENT-FOCUSED critiques. "
            f"Does the answer fully address the patient's concerns as expressed in the query? "
            f"Output ONLY the critique."
        )
        return self._call_deepseek_llm("Patient Criterion Assessment", prompt, max_tokens=8192)

    def compute_textual_gradient_for_answer_kc(self, answer_a: str, context_c: str, critiques_ckc: str) -> str:
        prompt = (
            f"You are an expert medical editor. Your task is to provide actionable instructions to refine a given medical answer based on specific critiques related to knowledge context.\n\n"
            f"Original Answer (A):\n\"{answer_a}\"\n\n"
            f"Relevant Medical Knowledge Context (C) (excerpt):\n\"{context_c[:4000]}...\"\n\n"
            f"KNOWLEDGE-FOCUSED Critiques on the Original Answer:\n\"{critiques_ckc}\"\n\n"
            f"Based ONLY on the provided critiques and referring to the knowledge context, explain step-by-step how to revise the Original Answer to address these critiques. "
            f"The explanation should focus on specific, targeted revisions that address the critiques while maintaining the overall structure and valid information in the original answer as much as possible. "
            f"Your output should be actionable instructions for improving the answer. Do not write the revised answer itself. Output ONLY the instructions."
        )
        return self._call_deepseek_llm("Textual Gradient (Knowledge Critiques w.r.t. Answer)", prompt, max_tokens=8192)

    def compute_textual_gradient_for_answer_pc(self, answer_a: str, query_q: str, critiques_cpc: str) -> str:
        prompt = (
            f"You are an expert medical editor. Your task is to provide actionable instructions to refine a given medical answer based on specific critiques related to the patient's query.\n\n"
            f"Original Answer (A):\n\"{answer_a}\"\n\n"
            f"Original Patient Query (q):\n\"{query_q}\"\n\n"
            f"PATIENT-FOCUSED Critiques on the Original Answer:\n\"{critiques_cpc}\"\n\n"
            f"Based ONLY on the provided critiques and referring to the patient's query, explain step-by-step how to revise the Original Answer to address these critiques. "
            f"The explanation should focus on specific, targeted revisions that address the critiques while maintaining the overall structure and valid information in the original answer as much as possible. "
            f"Your output should be actionable instructions for improving the answer. Do not write the revised answer itself. Output ONLY the instructions."
        )
        return self._call_deepseek_llm("Textual Gradient (Patient Critiques w.r.t. Answer)", prompt, max_tokens=8192)

    def compute_textual_gradient_for_prompt_kc(self, prompt_p: str, answer_a: str, grad_answer_kc: str) -> str:
        prompt = (
            f"You are an AI assistant that refines prompts for a medical question-answering system.\n\n"
            f"The Original Prompt (P) used was (excerpt):\n\"\"\"\n{prompt_p[:2000]}...\n\"\"\"\n\n"
            f"The answer A generated using P was:\n\"\"\"\n{answer_a}\n\"\"\"\n\n"
            f"The Answer (A) requires revisions from a KNOWLEDGE perspective, as suggested by the following feedback on A:\n\"\"\"\n{grad_answer_kc}\n\"\"\"\n\n"
            f"Your task: Explain precisely how to modify or rewrite the ORIGINAL PROMPT (P) to create an improved P. "
            f"This P should better guide the LLM to refine an answer like A, addressing the KNOWLEDGE-FOCUSED issues identified in the feedback. "
            f"Provide clear, actionable advice on prompt improvement. Output ONLY the advice."
        )
        return self._call_deepseek_llm("Textual Gradient (Knowledge Critiques w.r.t. Prompt)", prompt, max_tokens=8192)

    def compute_textual_gradient_for_prompt_pc(self, prompt_p: str, answer_a: str, grad_answer_pc: str) -> str:
        prompt = (
            f"You are an AI assistant that refines prompts for a medical question-answering system.\n\n"
            f"The Original Prompt (P) used was (excerpt):\n\"\"\"\n{prompt_p[:2000]}...\n\"\"\"\n\n"
            f"The answer A generated using P was:\n\"\"\"\n{answer_a}\n\"\"\"\n\n"
            f"The Answer (A) requires revisions from a PATIENT QUERY perspective, as suggested by the following feedback on A:\n\"\"\"\n{grad_answer_pc}\n\"\"\"\n\n"
            f"Your task: Explain precisely how to modify or rewrite the ORIGINAL PROMPT (P) to create an improved P. "
            f"This P should better guide the LLM to refine an answer like A, addressing the PATIENT-FOCUSED issues identified in grad_answer_pc. "
            f"Provide clear, actionable advice on prompt improvement. Output ONLY the advice."
        )
        return self._call_deepseek_llm("Textual Gradient (Patient Critiques w.r.t. Prompt)", prompt, max_tokens=8192)

    def textual_gradient_descent_step(self, current_prompt_p: str, grad_prompt_kc: str, grad_prompt_pc: str, query_q: str, initial_answer_example: str, answer_a: str) -> str:
        prompt = (
            f"You are an AI assistant tasked with refining a medical consultation prompt.\n\n"
            f"The Original Prompt (P_t) to be improved is (excerpt):\n\"\"\"\n{current_prompt_p[:2000]}...\n\"\"\"\n\n"
            f"The Original Answer generated by P_t is: \n\"\"\"\n{answer_a}...\n\"\"\"\n\n"
            f"We need to create an Updated Prompt (P) based on the following feedback, which aims to make P more effective for refining answers like A.\n"
            f"Feedback derived from KNOWLEDGE critiques (grad_prompt_kc):\n\"\"\"\n{grad_prompt_kc}\n\"\"\"\n"
            f"Feedback derived from PATIENT QUERY critiques (grad_prompt_pc):\n\"\"\"\n{grad_prompt_pc}\n\"\"\"\n\n"
            f"Original Patient Query (q) for context: {query_q[:500]}...\n"
            f"Example of an answer (A) that P_t was used to refine: {initial_answer_example[:500]}...\n\n"
            f"Your task: Generate the Updated Prompt (P).\n"
            f"This P must be a complete set of instructions and context. When P is later combined with a new 'current answer (A)', "
            f"the LLM receiving (P + actual text of A) should be guided to:\n"
            f"1. Produce a refined version of that 'current answer (A)'.\n"
            f"2. Implicitly incorporate the improvements suggested by the feedback (grad_prompt_kc, grad_prompt_pc) through your new instructions in P.\n"
            f"3. Ensure the refined answer directly addresses the patient's query, is factually sound, clear, and empathetic.\n"
            f"4. Output ONLY the refined medical answer itself, without any conversational preamble, meta-commentary, self-correction notes, or any text other than the refined answer.\n\n"
            f"Focus on making P a robust set of instructions for the refinement task.\n\n"
            f"Output ONLY the Updated Prompt (P)."
        )
        return self._call_deepseek_llm("Prompt Update (Textual Gradient Descent Step)", prompt, max_tokens=8192)

    def run(self, stage1_output: dict) -> tuple:
        self.history = []
        query_q = stage1_output['dialogue']
        knowledge_base_text = stage1_output['related_knowledge']
        retrieved_cases_p = stage1_output['similar_patients']
        answer_to_refine_initially = stage1_output['generated_answer']
        iteration_outputs = []
        self._log("--- Starting Med-TextGrad Optimization ---")
        self._log(f"Original Patient Query (q): {query_q}")
        self._log(f"Initial Answer to Refine (from Stage 1):\n{answer_to_refine_initially[:500]}...\n")
        context_c = self._aggregate_context(knowledge_base_text, retrieved_cases_p)
        initial_prompt_template_p0 = (
            f"You are a helpful medical consultation AI. Your task is to REFINE the 'current answer (A)' (which will be provided at the end of this prompt) "
            f"by critically evaluating it against the patient's query and the supporting context. "
            f"The goal is to improve the 'current answer (A)'s accuracy, completeness, and relevance to the patient's query, "
            f"while preserving its valid core information and avoiding unnecessary deviations from its original intent.\n\n"
            f"Patient Query (q):\n{query_q}\n\n"
            f"Supporting Context (C) (excerpt for guiding refinement):\n\"\"\"\n{context_c[:10000]}\n\"\"\"\n\n"
            f"Instructions for refinement:\n"
            f"1. Carefully read the 'Patient Query (q)' and the 'Supporting Context (C)'.\n"
            f"2. Critically evaluate the 'current answer (A)' (provided below) against this information.\n"
            f"3. Generate an improved and refined version of the 'current answer (A)'.\n"
            f"4. Focus on addressing any shortcomings in the 'current answer (A)' regarding accuracy, completeness, clarity, and direct relevance to the patient's query.\n"
            f"5. Ensure your refined answer is factually sound based on the context, empathetic, and easy for a patient to understand.\n"
            f"6. IMPORTANT: Your output must be ONLY the refined medical answer itself. Do not include any preamble, conversational phrases, meta-commentary, or any text other than the refined answer."
        )
        prompt_pt = initial_prompt_template_p0
        self._log(f"\n--- Initial Guiding Prompt Template (P_0) (First 500 chars) ---\n{prompt_pt[:500]}...\n---")
        current_answer_at = answer_to_refine_initially
        initial_medtextgrad_refined_a0 = "Error: MedTextGrad did not produce an initial refined answer."
        for t in range(self.num_iterations):
            self._log(f"\n\n--- Iteration {t+1}/{self.num_iterations} ---")
            self._log(f"Current Prompt Template (P_{t}) (First 500 chars):\n{prompt_pt[:500]}...")
            self._log(f"Current Answer (A_{t}) to refine (First 500 chars):\n{current_answer_at[:500]}...")
            current_iteration = {
                "iteration": t+1,
                "prompt_template": prompt_pt,
                "input_answer": current_answer_at
            }
            refined_answer = self.generator_agent(prompt_pt, current_answer_at)
            current_iteration["refined_answer"] = refined_answer
            if "[LLM Call Failed" in refined_answer or "[Unexpected LLM Call Error" in refined_answer or not refined_answer.strip():
                self._log(f"ERROR: Generation/Refinement failed in iteration {t+1}. Stopping optimization.")
                if not refined_answer.strip() and not ("[LLM Call Failed" in refined_answer or "[Unexpected LLM Call Error" in refined_answer):
                    refined_answer = "[LLM produced empty answer]"
                if t == 0:
                    initial_medtextgrad_refined_a0 = refined_answer
                current_iteration["error"] = "Generation/Refinement failed"
                iteration_outputs.append(current_iteration)
                break
            current_answer_at = refined_answer
            self._log(f"Generated/Refined Answer (A) at Iteration {t} (First 500 chars):\n{current_answer_at[:500]}...")
            if t == 0:
                initial_medtextgrad_refined_a0 = current_answer_at
            critiques_kc_t = self.knowledge_criterion_agent(current_answer_at, context_c)
            critiques_pc_t = self.patient_criterion_agent(current_answer_at, query_q)
            current_iteration["knowledge_critiques"] = critiques_kc_t
            current_iteration["patient_critiques"] = critiques_pc_t
            grad_ans_kc_t = self.compute_textual_gradient_for_answer_kc(current_answer_at, context_c, critiques_kc_t)
            grad_ans_pc_t = self.compute_textual_gradient_for_answer_pc(current_answer_at, query_q, critiques_pc_t)
            current_iteration["gradient_answer_knowledge"] = grad_ans_kc_t
            current_iteration["gradient_answer_patient"] = grad_ans_pc_t
            grad_prompt_kc_t = self.compute_textual_gradient_for_prompt_kc(prompt_pt, current_answer_at, grad_ans_kc_t)
            grad_prompt_pc_t = self.compute_textual_gradient_for_prompt_pc(prompt_pt, current_answer_at, grad_ans_pc_t)
            current_iteration["gradient_prompt_knowledge"] = grad_prompt_kc_t
            current_iteration["gradient_prompt_patient"] = grad_prompt_pc_t
            if t < self.num_iterations - 1:
                prompt_pt_plus_1 = self.textual_gradient_descent_step(
                    prompt_pt, grad_prompt_kc_t, grad_prompt_pc_t, query_q, answer_to_refine_initially, current_answer_at
                )
                current_iteration["updated_prompt_template"] = prompt_pt_plus_1
                if "[LLM Call Failed" in prompt_pt_plus_1 or "[Unexpected LLM Call Error" in prompt_pt_plus_1 or not prompt_pt_plus_1.strip():
                    self._log(f"ERROR: Failed to update prompt in iteration {t+1}. Using previous prompt for next iteration.")
                    if not prompt_pt_plus_1.strip() and not ("[LLM Call Failed" in prompt_pt_plus_1 or "[Unexpected LLM Call Error" in prompt_pt_plus_1):
                        self._log("Reason: LLM produced empty prompt.")
                    current_iteration["prompt_update_error"] = True
                else:
                    prompt_pt = prompt_pt_plus_1
            else:
                self._log("Final iteration of loop. No further prompt update within the loop.")
            iteration_outputs.append(current_iteration)
        final_refined_answer_a_t = current_answer_at
        self._log(f"\n--- Med-TextGrad Optimization Finished ---")
        self._log(f"Final Refined Answer (AT) (First 500 chars):\n{final_refined_answer_a_t[:500]}...")
        return initial_medtextgrad_refined_a0, final_refined_answer_a_t, self.history, iteration_outputs

def calculate_evaluation_metrics(candidate: str, reference: str, lang="en") -> dict:
    metrics = {}
    try:
        nltk.word_tokenize("test")
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        meteor_score([["test"]], ["test"])
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    if not candidate or not isinstance(candidate, str) or not reference or not isinstance(reference, str) or candidate.startswith("[LLM") or candidate.startswith("Error:"):
        error_val = "N/A (input error or generation failure)"
        return {
            'ROUGE-L (F1)': error_val, 'BLEU-1': error_val, 'BLEU-2': error_val,
            'BLEU-3': error_val, 'BLEU-4': error_val, 'METEOR': error_val,
            'BERTScore (P)': error_val, 'BERTScore (R)': error_val, 'BERTScore (F1)': error_val
        }
    ref_tokens_list = nltk.word_tokenize(reference)
    can_tokens_list = nltk.word_tokenize(candidate)
    if not can_tokens_list:
        error_val = "N/A (empty candidate)"
        return {
            'ROUGE-L (F1)': error_val, 'BLEU-1': error_val, 'BLEU-2': error_val,
            'BLEU-3': error_val, 'BLEU-4': error_val, 'METEOR': error_val,
            'BERTScore (P)': error_val, 'BERTScore (R)': error_val, 'BERTScore (F1)': error_val
        }
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        metrics['ROUGE-L (F1)'] = scores['rougeL'].fmeasure
    except Exception as e:
        print(f"Error calculating ROUGE-L: {e}")
        metrics['ROUGE-L (F1)'] = "Error"
    chencherry = SmoothingFunction()
    try:
        for n in range(1, 5):
            if len(can_tokens_list) >= n and len(ref_tokens_list) >= n:
                weights = tuple(1/n for _ in range(n))
                metrics[f'BLEU-{n}'] = sentence_bleu([ref_tokens_list], can_tokens_list, weights=weights, smoothing_function=chencherry.method1)
            else:
                metrics[f'BLEU-{n}'] = 0.0
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        for n in range(1, 5):
            metrics[f'BLEU-{n}'] = "Error"
    try:
        metrics['METEOR'] = meteor_score([" ".join(ref_tokens_list)], " ".join(can_tokens_list))
    except Exception as e:
        print(f"Error calculating METEOR: {e}. Make sure NLTK's 'wordnet' and 'omw-1.4' are downloaded.")
        metrics['METEOR'] = "Error"
    try:
        P, R, F1 = bert_score_calculate([candidate], [reference], lang=lang, verbose=False, model_type='bert-base-multilingual-cased', idf=False)
        metrics['BERTScore (P)'] = P.mean().item()
        metrics['BERTScore (R)'] = R.mean().item()
        metrics['BERTScore (F1)'] = F1.mean().item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        metrics['BERTScore (P)'] = "Error"
        metrics['BERTScore (R)'] = "Error"
        metrics['BERTScore (F1)'] = "Error"
    return metrics

if __name__ == '__main__':
    DEEPSEEK_API_KEY_FROM_ENV = "YOUR-API-KEY"
    DEEPSEEK_BASE_URL_FROM_ENV = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL_FROM_ENV = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    if not DEEPSEEK_API_KEY_FROM_ENV:
        print("Error: DEEPSEEK_API_KEY environment variable not set or key is missing.")
        exit(1)
    json_file_path = "/apdcephfs_qy3/share_301997302/louisyuzhao/buddy1/coltonlu/PatientIndexLibrary_EN/GeneratedResults/COD/qwen/rag_answers_evaluation_qwen_copy.json"
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data or "results" not in data or not isinstance(data["results"], list) or not data["results"]:
            print(f"Error: No results found or results is not a list in {json_file_path}")
            exit(1)
        print(f"Successfully loaded {len(data['results'])} results from JSON file")
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit(1)
    all_results_output = []
    print("--- Med-TextGrad Optimizer ---")
    print(f"Using Model: {DEEPSEEK_MODEL_FROM_ENV} at {DEEPSEEK_BASE_URL_FROM_ENV}")
    print(f"Number of Iterations: {DEFAULT_ITERATIONS}\n")
    optimizer = MedTextGradOptimizer(
        api_key=DEEPSEEK_API_KEY_FROM_ENV,
        base_url=DEEPSEEK_BASE_URL_FROM_ENV,
        model_name=DEEPSEEK_MODEL_FROM_ENV,
        num_iterations=DEFAULT_ITERATIONS,
        verbose=True
    )
    for idx, result_item in enumerate(data["results"]):
        if idx <= 10 or idx > 50:
            continue
        print(f"\n\nProcessing result {idx+1}/{len(data['results'])}: ID {result_item.get('id', 'unknown')}")
        required_keys = ['dialogue', 'related_knowledge', 'similar_patients', 'generated_answer', 'reference_answer']
        if not all(key in result_item for key in required_keys):
            print(f"Skipping result ID {result_item.get('id', 'unknown')} due to missing required keys.")
            error_entry = {
                "id": result_item.get("id", "unknown"),
                "error_message": "Skipped due to missing required keys in input JSON item.",
                "answers": {}, "metrics": {}, "history_log": [], "iteration_outputs": []
            }
            all_results_output.append(error_entry)
            with open(f"medtextgrad_error_{result_item.get('id', idx)}.json", "w", encoding="utf-8") as f_err:
                json.dump(error_entry, f_err, ensure_ascii=False, indent=2)
            continue
        stage1_example_output = result_item
        current_result_data = {}
        initial_A0_refined, final_AT, history_log, iteration_outputs = "[Optimization Not Run]", "[Optimization Not Run]", [], []
        try:
            initial_A0_refined, final_AT, history_log, iteration_outputs = optimizer.run(stage1_example_output)
        except ValueError as e:
            print(f"Configuration Error for ID {result_item.get('id', 'unknown')}: {e}")
            initial_A0_refined, final_AT = f"[Config Error: {e}]", f"[Config Error: {e}]"
        except OpenAIError as e:
            print(f"An API error occurred during Med-TextGrad for ID {result_item.get('id', 'unknown')}: {e}")
            initial_A0_refined, final_AT = f"[API Error: {e}]", f"[API Error: {e}]"
        except Exception as e:
            print(f"An unexpected error occurred during Med-TextGrad for ID {result_item.get('id', 'unknown')}: {e}")
            initial_A0_refined, final_AT = f"[Unexpected Error: {e}]", f"[Unexpected Error: {e}]"
        reference = stage1_example_output['reference_answer']
        stage1_original_answer = stage1_example_output['generated_answer']
        answers_to_evaluate = {
            "Stage 1 Original Answer (Input to Med-TextGrad)": stage1_original_answer,
            "Med-TextGrad Initial Refinement (A0_refined)": initial_A0_refined,
            "Med-TextGrad Final Refined Answer (AT)": final_AT
        }
        print("\n===================================================")
        print(f"           RESULTS & EVALUATION SUMMARY - ID {result_item.get('id', 'unknown')}          ")
        print("===================================================")
        print(f"\nREFERENCE ANSWER:\n-----------------\n{reference}\n")
        result_metrics = {}
        for name, candidate_answer in answers_to_evaluate.items():
            print(f"\nCANDIDATE: {name}\n-----------------")
            candidate_answer_str = str(candidate_answer) if candidate_answer is not None else "[NoneType Answer]"
            if not (
                candidate_answer_str.startswith("[LLM")
                or candidate_answer_str.startswith("[API Error")
                or candidate_answer_str.startswith("[Config Error")
                or candidate_answer_str.startswith("[Unexpected Error")
                or candidate_answer_str.startswith("[Optimization Not Run]")
                or candidate_answer_str == "[NoneType Answer]"
            ):
                print(f"Answer Text (first 500 chars):\n{candidate_answer_str[:500]}...\n")
                print("Metrics (vs. Reference):")
                metrics = calculate_evaluation_metrics(candidate_answer_str, reference, lang="en")
                result_metrics[name] = metrics
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric_name:<18}: {value:.4f}")
                    else:
                        print(f"  {metric_name:<18}: {value}")
            else:
                print(f"Answer Text: {candidate_answer_str}\n")
                print("Metrics: Not calculated due to error/issue in answer generation.")
                result_metrics[name] = {"error": candidate_answer_str, "detail": "Metrics not calculated due to generation failure or error state."}
            print("---------------------------------------------------")
        current_result_data = {
            "id": result_item.get("id", "unknown"),
            "query": stage1_example_output['dialogue'],
            "reference_answer": reference,
            "answers": answers_to_evaluate,
            "metrics": result_metrics,
            "history_log": history_log,
            "iteration_outputs": iteration_outputs
        }
        all_results_output.append(current_result_data)
        with open(f"medtextgrad_result_{result_item.get('id', idx)}.json", "w", encoding="utf-8") as f:
            json.dump(current_result_data, f, ensure_ascii=False, indent=2)
    with open("medtextgrad_all_results_revised.json", "w", encoding="utf-8") as f:
        json.dump(all_results_output, f, ensure_ascii=False, indent=2)
    print("\nAll results processed and saved to medtextgrad_all_results_revised.json")