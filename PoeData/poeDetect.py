import pandas as pd
import os
import re



from transformers import BertTokenizer, BertForSequenceClassification
import torch


def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip().lower()  # Lowercase and strip leading/trailing whitespace
    return text

def load_classifier():
    # Load the trained model and tokenizer
    model = BertForSequenceClassification.from_pretrained('./poe_classifier')
    tokenizer = BertTokenizer.from_pretrained('./poe_classifier')
    return model, tokenizer

def classify_text(text, model, tokenizer):

    # Preprocess and tokenize the text
    inputs = tokenizer(text, return_tensors='pt')

    # Perform classification
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()

    # Interpret the predicted class
    if predicted_class == 1:
        return True
    else:
        return False

if __name__ == "__main__":
    text = """Those petty wrongs that liberty commits,
When I am sometime absent from thy heart,
Thy beauty and thy years full well befits,
For still temptation follows where thou art."""
    model, tokenizer = load_classifier()
    preprocessed_text = preprocess_text(text)
    classification = classify_text(text, model, tokenizer)
    print(f"The text is classified as: {classification}")