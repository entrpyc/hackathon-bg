const express = require('express');
const { predict } = require('./python-scripts');

const app = express();

app.get('/predict', async (req, res) => {
  const { temperature, rainfall, humidity, tire_make, tire_model } = req.query;

  if (!temperature || !rainfall || !humidity || !tire_make || !tire_model) {
    return res.status(400).send({
      error: 'All of the following parameters are required: temperature, rainfall, humidity, tire_make, tire_model'
    });
  }

  const result = await predict({ temperature, rainfall, humidity, tire_make, tire_model });

  res.send({ result: Number(result) });
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
