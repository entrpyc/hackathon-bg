import sys
import os
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
from pathlib import Path
from model import DistilBertForRegression

script_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer_path = file1_path = os.path.join(script_dir, 'tokenizer/')
scaler_path = os.path.join(script_dir, 'scaler.joblib')
model_path = os.path.join(script_dir, 'distilbert_regression_model.pth')

tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

scaler = joblib.load(scaler_path)

model = DistilBertForRegression()
model.load_state_dict(torch.load(model_path))
model.eval()

def predict_single_example(example):
    inputs = tokenizer.encode_plus(
        f"{example[0]} {example[1]} {example[2]} {example[3]} {example[4]} {example[5]} {example[6]} {example[7]} {example[8]} {example[9]} {example[10]} {example[11]} {example[12]} {example[13]} {example[14]} {example[15]} {example[16]} {example[17]} {example[18]} {example[19]} {example[20]}",
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        return_attention_mask=True
    )
    input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask).squeeze()
    return scaler.inverse_transform(output.cpu().numpy().reshape(-1, 1))[0][0]

if __name__ == "__main__":
    if len(sys.argv) != 22:
        print("Usage: python predict.py <thread_depth> <tire_type> <tire_width> <tire_diameter> <tire_ratio> <car_weight> <pressure_checks_frequency> <city_avg_speed> <outside_city_avg_speed> <driving_style> <paved_road> <offroad> <paved_road_quality> <offroad_quality> <min_temperature> <max_temperature> <avg_temperature> <driving_frequency> <car_extra_load_weight> <tire_age> <distance_driven_with_tires>")
        sys.exit(1)
    
    example_data = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14], sys.argv[15], sys.argv[16], sys.argv[17], sys.argv[18], sys.argv[19], sys.argv[20], sys.argv[21]]
    predicted_tire_life = predict_single_example(example_data)
    print(predicted_tire_life)