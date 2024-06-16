# Tire wear prediction model

This is a machine learning model with the goal to estimate the remaining distance car tires can be used for before needing to be replaced. 

## Installing dependencies

Before running the model the dependencies need to be installed:

### Creating a virtual environment

To create a new python virtual environment run:

`python -m venv ./venv`

Then to activate it:

`source venv/bin/activate`

### Installing the required dependencies

To install the required dependencies run:

`pip install -r requirements.txt`

## Running the model

To train the model against the dataset available in the repo run:

`python train.py`

To make an individual prediction after training the model, run:

```
python predict.py <thread_depth> <tire_type> <tire_width> <tire_diameter> <tire_ratio> <car_weight> <pressure_checks_frequency> <city_avg_speed> <outside_city_avg_speed> <driving_style> <paved_road> <offroad> <paved_road_quality> <offroad_quality> <min_temperature> <max_temperature> <avg_temperature> <driving_frequency> <car_extra_load_weight> <tire_age> <distance_driven_with_tires>`
```

Making predictions is also possible through an api:

To start the api on port 3000 run `python api.py`

To make predictions you can use the `GET /predict` endpoint, which takes the arguments as query params:

```
curl http://localhost:3000/predict?thread_depth=4&tire_type=summer&tire_width=15&tire_diameter=18&tire_ratio=65&car_weight=2300&pressure_checks_frequency=6&city_avg_speed=50&outside_city_avg_speed=90&driving_style=2&paved_road=8&offroad=1&paved_road_quality=6&offroad_quality=3&min_temperature=-10&max_temperature=36&avg_temperature=29&driving_frequency=4&car_extra_load_weight=0&tire_age=0&distance_driven_with_tires=20000
```
