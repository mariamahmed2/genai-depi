from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('mnist_cnn_model.keras')

# Initialize FastAPI app
app = FastAPI()

# Define request model
class ImageRequest(BaseModel):
    image: list

# Define predict endpoint
@app.post("/predict")
def predict(request: ImageRequest):
    image = np.array(request.image).reshape((1, 28, 28, 1))
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    return {"predicted_label": int(predicted_label)}
