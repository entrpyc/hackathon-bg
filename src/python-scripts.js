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

const predict = async ({ temperature, rainfall, humidity, tire_make, tire_model }) => {
  const scriptPath = path.join(__dirname, '../model/venv/bin/python');
  const filePath = path.join(__dirname, '../model/predict.py');

  try {

    return await invokeScript(scriptPath, [filePath, temperature, rainfall, humidity, tire_make, tire_model]);
  } catch (error) {
    console.log(error)
    return error;
  }
}

module.exports = { predict }