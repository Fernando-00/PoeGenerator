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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, GPT2Config, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from torch.utils.data import Dataset

def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        super(Adapter, self).__init__()
        self.linear1 = nn.Linear(input_dim, bottleneck_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(bottleneck_dim, input_dim)
        
    def forward(self, x):
        down = self.linear1(x)
        activated = self.relu(down)
        up = self.linear2(activated)
        return x + up
    
class GPT2WithAdaptersAndLayerNorm(GPT2LMHeadModel):
    def __init__(self, config):
        super(GPT2WithAdaptersAndLayerNorm, self).__init__(config)
        self.adapters = nn.ModuleList([Adapter(config.hidden_size) for _ in range(config.n_layer)])
        self.pre_layer_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs.last_hidden_state
        
        for adapter in self.adapters:
            hidden_states = adapter(hidden_states)
        
        hidden_states = self.pre_layer_norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        output = (lm_logits,) + transformer_outputs[1:]

        if loss is None:
            return output
        
        return ((loss,) + output)

# model.bert.encoder.layer[5] = CustomBertLayer(model.bert.encoder.layer[5])

def train_gpt2(dataset, tokenizer):
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    
    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2WithAdaptersAndLayerNorm(config)

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
        train_dataset=dataset,
        data_collator=data_collator
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
    # folder_path = "PoeData"
    # raw_text = load_poe_data(folder_path)
    # processed_text = preprocess_text(raw_text)
    
    # with open("processed_poe_data.txt", "w", encoding="utf-8") as file:
    #     file.write(processed_text)
        
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # dataset = load_dataset("processed_poe_data.txt", tokenizer)
    # train_gpt2(dataset, tokenizer)

    # print("loading models")
    model = GPT2LMHeadModel.from_pretrained("./gpt2-poe2")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-poe2")
    
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = """How can I then return in happy plight,"""
    prompt = preprocess_text(prompt)
    print("generating text")
    # a = evaluation(model)
    a = generate_text(prompt, model, tokenizer)
    print(a)
