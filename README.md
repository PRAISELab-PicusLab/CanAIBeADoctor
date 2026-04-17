# 🩺🤖  Can AI be a Doctor? **A Study of Empathy, Readability, and Alignment in Clinical LLMs**

---

## 🧠 Overview

As Large Language Models (LLMs) are increasingly deployed in healthcare, ensuring their **alignment with human values** such as **empathy**, **clarity**, and **semantic fidelity** becomes crucial.  
This project investigates whether LLMs can produce **patient-centered clinical communication**, comparing general-purpose and medical-specialized models on over **16,400 expert-annotated QA pairs**.

We introduce a comprehensive evaluation framework, test multiple LLM configurations, and propose a collaborative pipeline to enhance physician-authored responses.

---

## 💡 Research Questions

1. **Empathy:** Can LLMs convey emotional nuance as physicians do?
2. **Readability:** Are AI-generated answers more accessible than physician-authored ones?
3. **Prompting:** Does prompt engineering improve communication quality?
4. **Collaboration:** Can LLMs enhance existing medical responses?
5. **Value Alignment:** How do expert and patient evaluations differ in their preferences?

---

## 🏗️ Methodology

Our framework assesses three core dimensions:

| Dimension         | Metric(s)                             |
|-------------------|---------------------------------------|
| Semantic Fidelity | Cosine Similarity (MiniLM)            |
| Readability       | FKGL, Gunning Fog Index               |
| Empathy           | Sentiment + Emotion Classification    |

We evaluated:
- **General-Purpose LLM**: *Mixtral*
- **Medical-Fine-Tuned LLM**: *PALMYRA-Med*

Each under:
- `Base Prompt`: Standard formal medical tone
- `Empathy Prompt`: Empathy-enhanced, patient-centered
- `Caring Prompt`: LLM-refinement of physician-authored answers

The framework compares LLM-generated and physician-authored answers across semantic similarity, readability,
sentiment, and emotion. It includes both direct generation and LLM-based revision of expert responses, enabling evaluation of
AI models as autonomous communicators and collaborative assistants in clinical settings. 

![CAIbeaD_Workflow](./CAIbeaD_Workflow.png)

---

## 📊 Key Findings

| Configuration         | Empathy↑ | Readability↑ | Semantic Fidelity↑ |
|-----------------------|----------|--------------|--------------------|
| **LLM GP Empathy**    | ✅       | ✅✅        | ✅                |
| **LLM FT Caring**     | ✅✅     | ✅          | ✅✅              |
| **Physician Answers** | ❌       | ❌          | ✅✅              |

- Prompt engineering significantly reduces negative sentiment (↓ up to **20%**).
- LLMs improve readability by up to **2.5 grade levels**.
- Collaborative rewriting yields responses **preferred by both patients and experts**.

---

## 🧪 Dataset
We use a **curated subset** of the [**MedQuAD**](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) dataset:
- 16,400 QA pairs
- Sourced from NIH websites (e.g., MedlinePlus, cancer.gov)
- Covers 37 question types (symptoms, treatment, side effects)

---

## 🤖 Models & Prompts

| Variant              | Description                                           |
|----------------------|-------------------------------------------------------|
| `LLM GP`             | Mixtral + Base Prompt                                 |
| `LLM FT`             | PALMYRA-Med + Base Prompt                             |
| `LLM GP Empathy`     | Mixtral + Empathy Prompt                              |
| `LLM FT Empathy`     | PALMYRA-Med + Empathy Prompt                          |
| `LLM GP Caring`      | Mixtral rewrites physician answer                     |
| `LLM FT Caring`      | PALMYRA rewrites physician answer                     |

📎 Prompt templates are provided in the APPENDIX of the paper.

---

## 👥 Human Evaluation

We conducted **dual evaluations**:
- 🩺 **Expert Simulation** (via GPT-4o) → Precision & accuracy
- 🧑‍⚕️ **Patient Ratings** (n=30) → Trust, empathy, comprehensibility

**Top-rated variant overall**: `LLM FT Caring`  
**Most empathetic**: `LLM GP Empathy`  
**Best trade-off**: `LLM FT Empathy` (high accuracy, better readability)

---

## 📁 Repository Structure

```

CanAIWriteLikeCaringDoctor/
│
├── data/              # MedQuAD subset (processed)
├── analysis/          # Evaluation scripts (similarity, readability, emotion)
├── results/           # Figures, tables, evaluation metrics
└── README.md          # This file

```
---

We welcome contributions to improve our work! To contribute, simply open a pull request or report issues on our [issue tracker](https://github.com/PRAISELab-PicusLab/CanAIWriteLikeCaringDoctor/issues). We look forward to your improvements!

👨‍💻 This project was developed by Mariano Barone, Antonio Romano, Giuseppe Riccio, Marco Postiglione, and Vincenzo Moscato at *University of Naples Federico II* – [PRAISE Lab - PICUS](https://github.com/PRAISELab-PicusLab/)

---

## 🧭 License & Acknowledgments

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

---
