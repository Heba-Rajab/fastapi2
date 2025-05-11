
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Load pre-trained LSTM model
model = tf.keras.models.load_model("lstm_model.keras")

app = FastAPI(title="Predictive Maintenance API")

# Define input schema
class SensorData(BaseModel):
    Rotational_speed: float
    Torque: float
    Tool_wear: float
    Vibration: float
    Process_temperature: float

@app.post("/predict")
def predict(data: SensorData):
    try:
        # Preprocess input (reshape and convert to numpy array)
        input_data = np.array([[
            data.Rotational_speed,
            data.Torque,
            data.Tool_wear,
            data.Vibration,
            data.Process_temperature
        ]]).reshape((1, 1, 5))

        prediction = model.predict(input_data)
        failure_prob = float(prediction[0][0])

        return {"failure_probability": failure_prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
