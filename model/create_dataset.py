import pandas as pd
import numpy as np
import os

tier_types = ['summer', 'winter', 'all-season']

# Function to generate synthetic data
def generate_data(num_entries):
    np.random.seed(42)

    data = {
        'thread_depth': np.random.randint(0, 10, size=num_entries), # thread depth in mm
        'tire_type': np.random.choice(tier_types, size=num_entries), # summer, winter, all-season
        'tire_width': np.random.randint(8, 28, size=num_entries), # tire width in inches
        'tire_diameter': np.random.randint(13, 22, size=num_entries), # tire diameter in inches
        'tire_ratio': np.random.randint(30, 85, size=num_entries), # tire ratio in percentage
        'car_weight': np.random.randint(1000, 4000, size=num_entries), # car weight in kg
        'pressure_checks_frequency': np.random.randint(1, 10, size=num_entries), # pressure checks frequency score
        'city_avg_speed': np.random.randint(20, 60, size=num_entries), # city average speed in km/h
        'outside_city_avg_speed': np.random.randint(60, 200, size=num_entries), # outside city average speed in km/h
        'driving_style': np.random.randint(0, 10, size=num_entries), # driving style score - higher score means more aggressive driving
        'paved_road': np.random.randint(0, 10, size=num_entries), # paved road score - higher score means more driving on paved roads
        'offroad': np.random.randint(0, 10, size=num_entries), # offroad score - higher score means more driving on offroads
        'paved_road_quality': np.random.randint(0, 10, size=num_entries), # paved road quality score - higher score means better quality roads
        'offroad_quality': np.random.randint(0, 10, size=num_entries), # offroad quality score - higher score means better quality offroads
        'min_temperature': np.random.randint(-20, 20, size=num_entries), # minimum temperature in Celsius
        'max_temperature': np.random.randint(20, 50, size=num_entries), # maximum temperature in Celsius
        'avg_temperature': np.random.randint(0, 40, size=num_entries), # average temperature in Celsius
        'driving_frequency': np.random.randint(0, 10, size=num_entries), # driving frequency score - higher score means more frequent driving
        'car_extra_load_weight': np.random.randint(0, 300, size=num_entries), # extra load weight in kg
        'tire_age': np.random.randint(0, 5, size=num_entries), # tire age in years
        'distance_driven_with_tires': np.random.randint(0, 100000, size=num_entries), # distance driven with tires in km
        'tire_life_remaining': np.random.randint(0, 100000, size=num_entries) # tire life remaining in km
    }
    
    return pd.DataFrame(data)

# Generate the dataset
num_entries = 1000
df = generate_data(num_entries)

# Display the first few rows of the dataset
print(df.head())


script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = file1_path = os.path.join(script_dir, 'tire_life_dataset.csv')
# Save the dataset to a CSV file
df.to_csv(dataset_path, index=False)