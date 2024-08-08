from fastapi import FastAPI, Form, HTTPException
from utils import predict_new
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# laod .env
_ = load_dotenv(override=True)
SECRET_KEY = os.getenv('API_KEY_TOKEN')

app = FastAPI(title='Churn Prediction')


@app.get('/', tags=['General'])
def home():
    return {'Up & Running'}


# Pydantic class
class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float




@app.post('/predict', tags=['Churn'])
def churn_prediction(data: CustomerData):

    # if key_token not in [SECRET_KEY]:
    #     raise HTTPException(status_code=403, detail='You are not authorized to use this API')
         
    
    # call (predict_new) from utils.py
    case_pred = predict_new(data=data)

    return {f'Prediction is {case_pred}'}