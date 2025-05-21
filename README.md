# Towards Doctor-Like Reasoning: Medical RAG Fusing Knowledge with Patient Analogy through Textual Gradients

**DoctorRAG** is an advanced medical Retrieval-Augmented Generation (RAG) framework that emulates doctor-like reasoning by integrating explicit medical knowledge with experiential insights from real patient cases. **Med-TextGrad** is a multi-agent, iterative answer refinement process that further enhances the accuracy, relevance, and safety of generated responses.

## Figure

![image](https://github.com/YuxingLu613/DoctorRAG/blob/main/DoctorRAG%20Overview.png)

*Figure: Overview of the DoctorRAG framework, illustrating the integration of medical knowledge and patient analogical reasoning via Med-TextGrad.*


---

## Features

- **Hybrid Retrieval:** Combines knowledge base (medical expertise) and patient base (clinical experience) for context-rich answers.
- **Concept Tagging & Declarative Transformation:** Structures both queries and knowledge for precise, concept-aware retrieval.
- **Iterative Answer Optimization:** Med-TextGrad uses multi-agent textual gradients to iteratively refine answers.
- **Multilingual & Multitask:** Supports Chinese, English, and French datasets for diagnosis, QA, treatment recommendation, and text generation.
- **Plug-and-Play LLMs:** Compatible with OpenAI, DeepSeek, and other major LLM APIs.
- **Efficient Retrieval:** Uses FAISS for fast, scalable similarity search.

---

## Project Structure

```
DoctorRAG/
├── Datasets/           # Raw and processed datasets (CSV, by benchmark/language)
├── Knowledge_Base/     # Structured medical knowledge bases (multi-language, FAISS indices)
├── Patient_Base/       # Patient case data and FAISS indices
├── Outputs/            # Results and logs from experiments
├── Scripts/            # All experiment, training, and evaluation code
│   ├── DoctorRAG/      # Main DoctorRAG pipeline scripts
│   └── Med-TextGrad/   # Med-TextGrad iterative refinement and evaluation
├── Utils/              # Utility scripts (concept tagging, declarative transformation, etc.)
├── requirements.txt    # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- API access for OpenAI, DeepSeek, or other supported LLMs
- Sufficient disk space for datasets and FAISS indices

### Installation

1. **Clone the repository:**
   ```bash
   git clone [<https://github.com/YuxingLu613/DoctorRAG>](https://github.com/YuxingLu613/DoctorRAG)
   cd DoctorRAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys:**
   - Configure your OpenAI/DeepSeek API keys as required in the scripts (see comments in `Scripts/DoctorRAG/` and `Scripts/Med-TextGrad/`).

---

## Usage

### 1. Data Preparation

- Place your datasets in `Datasets/` and knowledge/patient bases in their respective folders.
- Each dataset/knowledge base should be organized by language and benchmark (see folder structure above).

### 2. Running DoctorRAG

- Use the scripts in `Scripts/DoctorRAG/` to run retrieval-augmented generation on your chosen dataset.
- Example:
  ```bash
  python Scripts/DoctorRAG/DDXPlus_EN_DD.py
  ```

### 3. Iterative Answer Refinement (Med-TextGrad)

- Use `Scripts/Med-TextGrad/Med_TextGrad.py` to iteratively refine answers.
- For pairwise evaluation, use `Scripts/Med-TextGrad/Pairwise-Rater.py`.

### 4. Outputs

- Results and logs are saved in `Outputs/` under the relevant method and experiment.

---

## Utilities

- `Utils/MedQA_Declarative_Sentence_*.py`: Convert knowledge chunks to declarative statements.
- `Utils/MedQA_Concept_Tagging_*.py`: Tag statements with medical concepts.
- `Utils/Knowledge_Base_Faiss.py`: Build and query FAISS indices for fast retrieval.

---

## Datasets

- **Datasets**: Multilingual, multitask medical QA and patient case datasets (see `Datasets/`).
- **Knowledge Bases**: Structured medical knowledge in English, Chinese, and French (see `Knowledge_Base/`).
- **Patient Bases**: De-identified patient records for experience-based retrieval (see `Patient_Base/`).

*Note: Due to size restrictions, some folders may contain placeholders.*

---

## Citation

If you use this code or data, please cite our paper (TBA):

---

## Contact

For questions or collaborations, please contact [yxlu0613@gmail.com].
