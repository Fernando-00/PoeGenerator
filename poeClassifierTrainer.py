import pandas as pd
import os
import re

def load_texts(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip().lower()  # Lowercase and strip leading/trailing whitespace
    return text

# Load and preprocess Poe texts
poe_folder_path = "PoeData"
poe_texts = load_texts(poe_folder_path)
poe_texts = [preprocess_text(text) for text in poe_texts]

# Load and preprocess non-Poe texts (replace 'NonPoeData' with actual folder)
non_poe_folder_path = "NonPoeData"
non_poe_texts = load_texts(non_poe_folder_path)
non_poe_texts = [preprocess_text(text) for text in non_poe_texts]

# Create DataFrame
data = {
    "text": poe_texts + non_poe_texts,
    "label": [1] * len(poe_texts) + [0] * len(non_poe_texts)
}
df = pd.DataFrame(data)
df.to_csv("poe_classifier_data.csv", index=False)


import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

# Load the dataset
dataset = load_dataset('csv', data_files='poe_classifier_data.csv')
dataset = dataset['train'].train_test_split(test_size=0.2)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format('torch')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define metrics
def compute_metrics(p):
    metric = load_metric('accuracy')
    return metric.compute(predictions=p.predictions.argmax(-1), references=p.label_ids)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./poe_classifier')
tokenizer.save_pretrained('./poe_classifier')


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('./poe_classifier')
tokenizer = BertTokenizer.from_pretrained('./poe_classifier')

# Load the dataset
dataset = load_dataset('csv', data_files='poe_classifier_data.csv')
dataset = dataset['train'].train_test_split(test_size=0.2)
tokenized_datasets = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True), batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format('torch')

# Evaluate the model
trainer = Trainer(
    model=model,
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=lambda p: {'accuracy': (p.predictions.argmax(-1) == p.label_ids).mean()},
)
results = trainer.evaluate()
print(results)