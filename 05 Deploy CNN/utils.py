import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import requests
import os
from PIL import Image
from io import BytesIO
from fastapi import HTTPException


# Load model
model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'dogs_cats.h5'))


# A folder for images to be fetched locally
IMAGES_FOLDER_PATH = os.path.join(os.getcwd(), 'images')
os.makedirs(IMAGES_FOLDER_PATH, exist_ok=True)
    


def fetch_image(url: str) -> str:

    # Construct the full path to save the image
    save_path = os.path.join(IMAGES_FOLDER_PATH, os.path.basename(url))
    
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Open the image from the response content
            img = Image.open(BytesIO(response.content))
            
            # Save the image to the specified path
            img.save(save_path)

            return save_path
        else:
            raise HTTPException(status_code=response.status_code, detail=f'Failed to fetch the image {url}')
  
    except Exception as e:
        raise HTTPException(status_code=422, detail=f'Failed to fetch the image {url}')
  


# Function to load and preprocess the image
def process_predict(img_path: str, target_size: tuple=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    y_pred = model.predict(img_array)
    y_pred = ['Dog' if val >= 0.5 else 'Cat' for val in y_pred][0]
    return y_pred


def remove_images():
    _ = [os.remove(img) for img in os.listdir(IMAGES_FOLDER_PATH)]
    return None
