import pandas as pd
import numpy as np
import os

# Define possible tire makes and models
tire_makes = ['Pirelli', 'Michelin', 'Goodyear']
tire_models = ['Winter1', 'Winter2', 'Summer1', 'Summer2']

# Function to generate synthetic data
def generate_data(num_entries):
    np.random.seed(42)
    
    data = {
        'temperature': np.random.randint(20, 40, size=num_entries),  # temperatures between 20 and 40 degrees Celsius
        'rainfall': np.random.randint(0, 10, size=num_entries),  # rainfall in mm
        'humidity': np.random.randint(50, 100, size=num_entries),  # humidity in percentage
        'tire_make': np.random.choice(tire_makes, size=num_entries),
        'tire_model': np.random.choice(tire_models, size=num_entries),
        'tire_life': np.random.randint(30000, 70000, size=num_entries)  # tire life in km
    }
    
    return pd.DataFrame(data)

# Generate the dataset
num_entries = 400
df = generate_data(num_entries)

# Display the first few rows of the dataset
print(df.head())


script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = file1_path = os.path.join(script_dir, 'tire_life_dataset.csv')
# Save the dataset to a CSV file
df.to_csv(dataset_path, index=False)