const path = require('path');
const { spawn } = require('child_process');

const invokeScript = (script, args) => new Promise((resolve, reject) => {
  const process = spawn(script, args);

  process.stdout.on('data', (data) => {
    resolve(data.toString());
  });

  process.stderr.on('data', (data) => {
    reject(data.toString());
  });
});

const predict = async ({
  thread_depth,
  tire_type,
  tire_width,
  tire_diameter,
  tire_ratio,
  car_weight,
  pressure_checks_frequency,
  city_avg_speed,
  outside_city_avg_speed,
  driving_style,
  paved_road,
  offroad,
  paved_road_quality,
  offroad_quality,
  min_temperature,
  max_temperature,
  avg_temperature,
  driving_frequency,
  car_extra_load_weight,
  tire_age,
  distance_driven_with_tires
}) => {
  const scriptPath = path.join(__dirname, '../model/venv/bin/python');
  const filePath = path.join(__dirname, '../model/predict.py');

  try {

    return await invokeScript(scriptPath, [
      filePath,
      thread_depth,
      tire_type,
      tire_width,
      tire_diameter,
      tire_ratio,
      car_weight,
      pressure_checks_frequency,
      city_avg_speed,
      outside_city_avg_speed,
      driving_style,
      paved_road,
      offroad,
      paved_road_quality,
      offroad_quality,
      min_temperature,
      max_temperature,
      avg_temperature,
      driving_frequency,
      car_extra_load_weight,
      tire_age,
      distance_driven_with_tires
    ]);
  } catch (error) {
    console.log(error)
    return error;
  }
}

module.exports = { predict }