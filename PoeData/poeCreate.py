import os
import re
import poeDetect as pdt
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

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

from torch.utils.data import Dataset

class PoemsDataset(Dataset):
    def __init__(self, texts, bert_tokenizer, gpt2_tokenizer, block_size=128):
        self.bert_tokenizer = bert_tokenizer
        self.gpt2_tokenizer = gpt2_tokenizer
        self.block_size = block_size

        # Tokenize all texts with BERT tokenizer
        self.bert_encodings = [self.bert_tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=block_size) for text in texts]
        
        # Tokenize all texts with GPT-2 tokenizer
        self.gpt2_encodings = [self.gpt2_tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=block_size) for text in texts]

    def __len__(self):
        return len(self.bert_encodings)

    def __getitem__(self, idx):
        bert_encoding = self.bert_encodings[idx]
        gpt2_encoding = self.gpt2_encodings[idx]
        return {
            'input_ids': bert_encoding['input_ids'].squeeze(),
            'attention_mask': bert_encoding['attention_mask'].squeeze(),
            'gpt2_input_ids': gpt2_encoding['input_ids'].squeeze(),
            'gpt2_attention_mask': gpt2_encoding['attention_mask'].squeeze(),
            'labels': gpt2_encoding['input_ids'].squeeze(),
        }
    


def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

import torch.nn as nn
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained BERT model and tokenizer

class BertGPT2Model(nn.Module):
    def __init__(self, bert_model, gpt2_model):
        super(BertGPT2Model, self).__init__()
        self.bert = bert_model
        self.gpt2 = gpt2_model
        
        # Project BERT's hidden state to GPT-2's hidden state size
        self.hidden_size_projection = nn.Linear(self.bert.config.hidden_size, self.gpt2.config.n_embd)
    
    def forward(self, input_ids, attention_mask=None, gpt2_input_ids=None, gpt2_attention_mask=None, labels=None):
        # BERT encoding
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        bert_hidden_states = bert_outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
    
        # print("bert_hidden_states shape:", bert_hidden_states.shape)
        
        # Project BERT hidden states to GPT-2 hidden size
        projected_hidden_states = self.hidden_size_projection(bert_hidden_states)
        # print("projected_hidden_states shape:", projected_hidden_states.shape)
        
        # Use BERT's projected hidden states as GPT-2's input embeddings
        gpt2_inputs_embeds = self.gpt2.transformer.wte(gpt2_input_ids) + projected_hidden_states
        # print("gpt2_inputs_embeds shape:", gpt2_inputs_embeds.shape)

        # GPT-2 decoding
        gpt2_outputs = self.gpt2(inputs_embeds=gpt2_inputs_embeds, attention_mask=gpt2_attention_mask, labels=labels)
        
        return gpt2_outputs
    
# model.bert.encoder.layer[5] = CustomBertLayer(model.bert.encoder.layer[5])

def train_gpt2(dataset):
    
    
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

    poems_dataset = PoemsDataset(dataset, bert_tokenizer, gpt2_tokenizer)

    model = BertGPT2Model(bert_model, gpt2_model)

    print("starting")

    training_args = TrainingArguments(
        output_dir="./gpt2-poe",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=10_000,
        logging_steps=100,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=poems_dataset,
        tokenizer=gpt2_tokenizer,
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
        
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # dataset = load_dataset("processed_poe_data.txt", tokenizer)
    train_gpt2(processed_text)

    print("loading models")
    model = GPT2LMHeadModel.from_pretrained("./gpt2-poe2")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-poe2")
    
    print("generating text")
    a = evaluation(model)
    print(a)
