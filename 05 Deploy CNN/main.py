from fastapi import FastAPI, BackgroundTasks
from utils import fetch_image, process_predict, remove_images


app = FastAPI(title='Dogs vs. Cats Classificaion Model', debug=True)


@app.get('/', tags=['General'])
async def home():
    return {'up & running'}



@app.post('/predict_class', tags=['Prediction'])
async def predict_class(url: str, background_tasks: BackgroundTasks):
    
    # Call functions from utils
    image_path = fetch_image(url=url)
    y_pred = process_predict(img_path=image_path)

    # Remove Images after getting response
    background_tasks.add_task(remove_images)

    return {'class': y_pred}