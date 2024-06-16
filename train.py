import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
from model import DistilBertForRegression


script_dir = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(script_dir, 'data.csv')
tokenizer_path = file1_path = os.path.join(script_dir, 'tokenizer/')
scaler_path = os.path.join(script_dir, 'scaler.joblib')
model_path = os.path.join(script_dir, 'distilbert_regression_model.pth')

df = pd.read_csv(dataset_path)

X = df[['thread_depth', 'tire_type', 'tire_width', 'tire_diameter', 'tire_ratio', 'car_weight', 'pressure_checks_frequency', 'city_avg_speed', 'outside_city_avg_speed', 'driving_style', 'paved_road', 'offroad', 'paved_road_quality', 'offroad_quality', 'min_temperature', 'max_temperature', 'avg_temperature', 'driving_frequency', 'car_extra_load_weight', 'tire_age', 'distance_driven_with_tires']]
y = df['tire_life_remaining'].values

scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1))

joblib.dump(scaler, scaler_path)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

tokenizer.save_pretrained(tokenizer_path)

def convert_to_tensors(data):
    input_ids = []
    attention_masks = []

    for index, row in data.iterrows():
        inputs = tokenizer.encode_plus(
            f"{row['thread_depth']} {row['tire_type']} {row['tire_width']} {row['tire_diameter']} {row['tire_ratio']} {row['car_weight']} {row['pressure_checks_frequency']} {row['city_avg_speed']} {row['outside_city_avg_speed']} {row['driving_style']} {row['paved_road']} {row['offroad']} {row['paved_road_quality']} {row['offroad_quality']} {row['min_temperature']} {row['max_temperature']} {row['avg_temperature']} {row['driving_frequency']} {row['car_extra_load_weight']} {row['tire_age']} {row['distance_driven_with_tires']}",
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])

    return torch.tensor(input_ids), torch.tensor(attention_masks)

X_train_ids, X_train_masks = convert_to_tensors(X_train)
X_val_ids, X_val_masks = convert_to_tensors(X_val)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

model = DistilBertForRegression()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

epochs = 10
batch_size = 50

for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_ids), batch_size):
        optimizer.zero_grad()
        batch_X_ids = X_train_ids[i:i+batch_size]
        batch_X_masks = X_train_masks[i:i+batch_size]
        batch_y = y_train[i:i+batch_size].squeeze()
        outputs = model(batch_X_ids, attention_mask=batch_X_masks).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_ids, attention_mask=X_val_masks).squeeze()
        val_loss = criterion(val_outputs, y_val.squeeze())
        val_mse = mean_squared_error(y_val.cpu().numpy(), val_outputs.cpu().numpy())
        print(f'Epoch [{epoch+1}/{epochs}], Validation MSE: {val_mse:.4f}')

torch.save(model.state_dict(), model_path)
print("Model saved successfully!")
