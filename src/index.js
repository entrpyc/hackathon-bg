const express = require('express');
const { predict } = require('./python-scripts');

const app = express();

const requiredFields = ['thread_depth', 'tire_type', 'tire_width', 'tire_diameter', 'tire_ratio', 'car_weight', 'pressure_checks_frequency', 'city_avg_speed', 'outside_city_avg_speed', 'driving_style', 'paved_road', 'offroad', 'paved_road_quality', 'offroad_quality', 'min_temperature', 'max_temperature', 'avg_temperature', 'driving_frequency', 'car_extra_load_weight', 'tire_age', 'distance_driven_with_tires'];

app.get('/predict', async (req, res) => {
  const params = req.query;

  for (const field of requiredFields) {
    if (!params[field]) {
      return res.status(400).send({
        error: `The following parameter is required: ${field}`
      });
    }
  }

  const result = await predict(params);

  res.send({ result: parseFloat(result) });
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
