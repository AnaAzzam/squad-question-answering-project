
#Question Answering System using HuggingFace Transformers on SQuAD Dataset

import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn.model_selection import train_test_split
from transformers import pipeline

# Step 1: Load the SQuAD dataset (v2.0)
with open('original_data/train-v2.0.json', 'r') as file:
    data = json.load(file)

contexts = []
questions = []
answers = []

# Step 2: Extract context, question, and answer from the dataset
for article in data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            if qa['is_impossible']:  # Skip unanswerable questions
                continue
            for answer in qa['answers']:
                contexts.append(context)
                questions.append(qa['question'])
                answers.append(answer['text'])

# Step 3: Create a DataFrame from the extracted data
df = pd.DataFrame({
    'context': contexts,
    'question': questions,
    'answer': answers
})

# Step 4: Sample 100 records for performance and simplicity
df = df.sample(100, random_state=42).copy()

# Step 5: Feature Engineering - Add new columns
df['context_len'] = df['context'].apply(len)
df['question_len'] = df['question'].apply(len)
df['answer_len'] = df['answer'].apply(len)
df['num_sentences'] = df['context'].apply(lambda x: len(re.findall(r'[.!?]', x)))
df['num_words'] = df['context'].apply(lambda x: len(x.split()))
df['has_wh_word'] = df['question'].apply(lambda q: int(bool(re.match(r'(?i)^(what|where|when|who|why|how)', q.strip()))))

# Step 6: Save preprocessed data
os.makedirs("preprocessed_data", exist_ok=True)
X_train, X_test, y_train, y_test = train_test_split(df['question'], df['answer'], test_size=0.3, random_state=42)

X_train.to_csv('preprocessed_data/X.csv', index=False)
y_train.to_csv('preprocessed_data/Y.csv', index=False)
X_test.to_csv('preprocessed_data/X_test.csv', index=False)
y_test.to_csv('preprocessed_data/Y_test.csv', index=False)

# Step 7: Save feature data for analysis
df[['context', 'question', 'answer', 'context_len', 'question_len', 'answer_len', 'num_sentences', 'num_words', 'has_wh_word']].to_csv('preprocessed_data/features.csv', index=False)

# Step 8: Create plots and save them
os.makedirs("Plots", exist_ok=True)

# Plot 1: Distribution of context length
plt.figure(figsize=(8, 4))
plt.hist(df['context_len'], bins=20, color='blue', edgecolor='black')
plt.title('Context Length Distribution')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Plots/context_length_distribution.png')
plt.close()

# Plot 2: Question length distribution
plt.figure(figsize=(8, 4))
plt.hist(df['question_len'], bins=20, color='green', edgecolor='black')
plt.title('Question Length Distribution')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Plots/question_length_distribution.png')
plt.close()

# Plot 3: Answer length distribution
plt.figure(figsize=(8, 4))
plt.hist(df['answer_len'], bins=20, color='orange', edgecolor='black')
plt.title('Answer Length Distribution')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Plots/answer_length_distribution.png')
plt.close()

# Plot 4: Number of sentences in context
plt.figure(figsize=(8, 4))
plt.hist(df['num_sentences'], bins=15, color='purple', edgecolor='black')
plt.title('Sentence Count in Context')
plt.xlabel('Number of Sentences')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Plots/context_sentence_count.png')
plt.close()

# Plot 5: Presence of WH-word in question
plt.figure(figsize=(6, 4))
df['has_wh_word'].value_counts().plot(kind='bar', color=['red', 'gray'])
plt.xticks([0, 1], ['No WH-word', 'Has WH-word'], rotation=0)
plt.title('Questions Starting with WH-Words')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Plots/wh_word_presence.png')
plt.close()

# Step 9: Load DistilBERT model and make predictions
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Step 10: Generate predictions using the QA model
predictions = []
for i in range(len(X_test)):
    question = X_test.iloc[i]
    context = df.iloc[i]['context']  # match the context to the question
    result = qa_model({
        'question': question,
        'context': context
    })
    predictions.append(result['answer'])

# Step 11: Save predictions to Results folder
os.makedirs("Results", exist_ok=True)
pd.DataFrame(predictions, columns=["Prediction"]).to_csv("Results/DistilBERT_model_predictions.csv", index=False)

print("Project executed successfully. Features, predictions, and plots are saved.")
