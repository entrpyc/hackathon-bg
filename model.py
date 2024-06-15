import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load and preprocess your dataset
df = pd.read_csv('tire_life_dataset.csv')

# Prepare data
X = df[['temperature', 'rainfall', 'humidity', 'tire_make', 'tire_model']]
y = df['tire_life'].values

# Normalize the target values
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize input data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

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

# Convert data to PyTorch tensors
def convert_to_tensors(data):
    input_ids = []
    attention_masks = []

    for index, row in data.iterrows():
        inputs = tokenizer.encode_plus(
            f"{row['temperature']} {row['rainfall']} {row['humidity']} {row['tire_make']} {row['tire_model']}",
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

# Initialize the model, optimizer, and loss function
model = DistilBertForRegression()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# Add a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Training loop
epochs = 10
batch_size = 32

for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_ids), batch_size):
        optimizer.zero_grad()
        batch_X_ids = X_train_ids[i:i+batch_size]
        batch_X_masks = X_train_masks[i:i+batch_size]
        batch_y = y_train[i:i+batch_size].squeeze()  # Remove the extra dimension
        outputs = model(batch_X_ids, attention_mask=batch_X_masks).squeeze()  # Ensure outputs are 1D
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    scheduler.step()

    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_ids, attention_mask=X_val_masks).squeeze()  # Ensure outputs are 1D
        val_loss = criterion(val_outputs, y_val.squeeze())  # Remove the extra dimension from targets
        val_mse = mean_squared_error(y_val.cpu().numpy(), val_outputs.cpu().numpy())
        print(f'Epoch [{epoch+1}/{epochs}], Validation MSE: {val_mse:.4f}')

# Example prediction
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
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask).squeeze()  # Ensure output is 1D
    return scaler.inverse_transform(output.cpu().numpy().reshape(-1, 1))[0][0]

example_data = [28, 2, 70, 'Pirelly', 'Summer2']
predicted_tire_life = predict_single_example(example_data)
print(f'Predicted Tire Life: {predicted_tire_life}')