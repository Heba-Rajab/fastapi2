AI-Powered-Predictive-Maintenance-for-Industrial-Equipment

This project is a machine learning API for predicting the probability of equipment failure using sensor readings.  
It uses **FastAPI** and an **LSTM neural network model** trained on time-series sensor data.

---

 Features

- Built with **FastAPI** and **TensorFlow**
- Trained LSTM model for failure prediction
- Hosted on **Azure Virtual Machine (Linux)**
- Live interactive Swagger UI for testing

---

 Deployment

The project is **deployed on an Azure Virtual Machine** using:
- Gunicorn + Uvicorn
- Nginx (as a reverse proxy)
- Public IP access for the live API

✅ **Live API Docs**:  
[http://20.174.2.78/docs](http://20.174.2.78/docs)

---

#### 🎬 Live Demo Video

Watch the full demo of the API running on Azure VM:

🔗 [Watch Demo on YouTube]([https://youtu.be/ABC123xyz](https://youtu.be/cBTI_ZQRm70))

### Sample Input:

```json
{
  "Rotational_speed": 1600.0,
  "Torque": 45.0,
  "Tool_wear": 120.0,
  "Vibration": 0.03,
  "Process_temperature": 308.0
}
---
 Sample Output:

{
  "failure_probability": 0.2649
}

