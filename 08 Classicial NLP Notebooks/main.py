from fastapi import FastAPI, Form
import os, joblib

app = FastAPI(title='Arabic Sentiment Analysis')
model = joblib.load(os.path.join(os.getcwd(), './model.pkl'))


@app.post('/', tags=['General'])
async def home():
    return {'message': 'up and running'}


@app.post('/predict', tags=['Sentiment Analysis'])
async def predict(text: str=Form(...)):
    y_pred = model.predict([text])
    return {'label': y_pred[0]}

    
    
