import sys
import os
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer_path = file1_path = os.path.join(script_dir, 'tokenizer/')
scaler_path = os.path.join(script_dir, 'scaler.joblib')
model_path = os.path.join(script_dir, 'distilbert_regression_model.pth')

# Define a simple model for regression using DistilBERT
class DistilBertForRegression(nn.Module):
    def __init__(self):
        super(DistilBertForRegression, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token's output
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

# Load the scaler
scaler = joblib.load(scaler_path)

# Load the model
model = DistilBertForRegression()
model.load_state_dict(torch.load(model_path))
model.eval()

def predict_single_example(example):
    inputs = tokenizer.encode_plus(
        f"{example[0]} {example[1]} {example[2]} {example[3]} {example[4]}",
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        return_attention_mask=True
    )
    input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask).squeeze()  # Ensure output is 1D
    return scaler.inverse_transform(output.cpu().numpy().reshape(-1, 1))[0][0]

if __name__ == "__main__":
    # Example data should be provided as command-line arguments
    if len(sys.argv) != 6:
        print("Usage: python predict.py <temperature> <rainfall> <humidity> <tire_make> <tire_model>")
        sys.exit(1)
    
    example_data = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]]
    predicted_tire_life = predict_single_example(example_data)
    print(predicted_tire_life)