import os
import re
import poeDetect as pdt

def load_poe_data(folder_path):
    text_data = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text_data += file.read() + " "
    return text_data

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip().lower()  # Lowercase and strip leading/trailing whitespace
    return text

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import BaseModelOutput


def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

import torch.nn as nn
from transformers import GPT2LMHeadModel



def train_gpt2(dataset, tokenizer):
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./gpt2-poe",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained("./gpt2-poe2")
    tokenizer.save_pretrained("./gpt2-poe2")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model, tokenizer, max_length='prompt'):
    model.eval()

    if (max_length == 'prompt'):
        max_length = int(len(prompt))
    inputs = tokenizer.encode(prompt, return_tensors='pt')    
    outputs = model.generate(
        input_ids=inputs, 
        num_beams=5, 
        max_new_tokens=max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=5.0,
        early_stopping=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text



def getFirstLines(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                first_line = file.readline().strip('\n')
                texts.append(first_line )
    return texts


def evaluation(model):
    model.eval()
    total, poelike = 0, 0
    cmodel, ctokenizer = pdt.load_classifier()
    non_poe_folder_path = "NonPoeData"
    non_poe_texts = getFirstLines(non_poe_folder_path)
    non_poe_texts = [preprocess_text(text) for text in non_poe_texts]
    for prompt in non_poe_texts:
        print(prompt)
        text = generate_text(prompt, model, tokenizer)
        ans = pdt.classify_text(text, cmodel, ctokenizer)
        if (ans):
            poelike += 1
        total += 1

    return poelike / total

if __name__ == "__main__":
    folder_path = "PoeData"
    raw_text = load_poe_data(folder_path)
    processed_text = preprocess_text(raw_text)
    
    with open("processed_poe_data.txt", "w", encoding="utf-8") as file:
        file.write(processed_text)
        
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = load_dataset("processed_poe_data.txt", tokenizer)
    train_gpt2(dataset, tokenizer)

    print("loading models")
    model = GPT2LMHeadModel.from_pretrained("./gpt2-poe2")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-poe2")
    
    print("generating text")
    a = evaluation(model)
    print(a)
