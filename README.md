# Question Answering System with SQuAD Dataset

## Project Title
**Building a Question Answering (QA) System using Hugging Face Transformers and the SQuAD Dataset**


## 1. Project Objective

The goal of this project is to build an intelligent system that can automatically extract answers from a paragraph based on natural language questions. We use a real-world benchmark dataset called **SQuAD (Stanford Question Answering Dataset)** and apply **pretrained BERT-based models** to answer the questions.

---

## 2. Dataset Description

- **Dataset Used**: [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/)
- **Source Format**: JSON
- Each sample contains:
  - A **context paragraph**
  - A **question**
  - One or more **answers** (text + start index)
- We use 100 samples from the training set (`train-v2.0.json`) for simplicity.

---

## 3. Tools and Libraries

- Python 3.10+
- `transformers` (Hugging Face)
- `pandas`
- `scikit-learn`
- `matplotlib`
- `torch`

Install using:

```bash
pip install -r requirements.txt
