from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from predict import predict_single_example

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
async def predict(
    thread_depth = Query(..., description="The depth of the tire's tread in mm"),
    tire_type = Query(..., description="The type of the tire (e.g. summer, winter, all-season)"),
    tire_width = Query(..., description="The width of the tire in inches"),
    tire_diameter = Query(..., description="The diameter of the tire in inches"),
    tire_ratio = Query(..., description="The ratio of the tire's height to its width"),
    car_weight = Query(..., description="The weight of the car in kg"),
    pressure_checks_frequency = Query(..., description="The frequency at which the tire pressure is checked (higher values indicate more frequent checks)"),
    city_avg_speed = Query(..., description="The average speed of the car in the city in km/h"),
    outside_city_avg_speed = Query(..., description="The average speed of the car outside the city in km/h"),
    driving_style = Query(..., description="The driving style of the car (higher values indicate a more aggressive driving style)"),
    paved_road = Query(..., description="The percentage of time the car is driven on paved roads"),
    offroad = Query(..., description="The percentage of time the car is driven off-road"),
    paved_road_quality = Query(..., description="The quality of the paved roads (higher values indicate better quality)"),
    offroad_quality = Query(..., description="The quality of the off-road conditions (higher values indicate better quality)"),
    min_temperature = Query(..., description="The minimum temperature the tire is exposed to in degrees Celsius"),
    max_temperature = Query(..., description="The maximum temperature the tire is exposed to in degrees Celsius"),
    avg_temperature = Query(..., description="The average temperature the tire is exposed to in degrees Celsius"),
    driving_frequency = Query(..., description="The frequency at which the car is driven (higher values indicate more frequent driving)"),
    car_extra_load_weight = Query(..., description="The weight of any extra load in the car in kg"),
    tire_age = Query(..., description="The age of the tire in years"),
    distance_driven_with_tires = Query(..., description="The distance driven with the tires in km")
):
    result = predict_single_example(
        [
            thread_depth, tire_type, tire_width, tire_diameter, tire_ratio, car_weight, pressure_checks_frequency, city_avg_speed, outside_city_avg_speed, driving_style, paved_road, offroad, paved_road_quality, offroad_quality, min_temperature, max_temperature, avg_temperature, driving_frequency, car_extra_load_weight, tire_age, distance_driven_with_tires
        ]
    )

    return {"result": float(result)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)