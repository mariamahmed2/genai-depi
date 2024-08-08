import joblib
import pandas as pd
import numpy as np
import os
from pydantic import BaseModel, Field


pipe = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'all_pipeline.pkl'))
model = joblib.load(os.path.join(os.getcwd(), 'artifacts', 'forest_tuned.pkl'))


dtypes = {
    "CreditScore": float,
    "Geography": str,
    "Gender": str,
    "Age": int,
    "Tenure": int,
    "Balance": float,
    "NumOfProducts": int,
    "HasCrCard": int,
    "IsActiveMember": int,
    "EstimatedSalary": float,
}



columns = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]



# Pydantic class
class CustomerData(BaseModel):
    CreditScore: float = Field(..., description='Credit score of the customer')
    Geography: str = Field(..., description='Geography')
    Gender: str = Field(..., description='Gender')
    Age: int = Field(..., description='Age of the customer')
    Tenure: int = Field(..., description='Number of years the customer has been with the bank')
    Balance: float = Field(..., description='Account balance')
    NumOfProducts: int = Field(..., description='Number of products the customer has')
    HasCrCard: int = Field(..., description='Does the customer have a credit card (1 for yes, 0 for no)')
    IsActiveMember: int = Field(..., description='Is the customer an active member (1 for yes, 0 for no)')
    EstimatedSalary: float = Field(..., description='Estimated salary of the customer')



def predict_new(data: CustomerData) -> str:

    # Concatenate the data
    input_data = np.array([data.CreditScore, data.Geography, data.Gender, data.Age, data.Tenure, data.Balance, 
                           data.NumOfProducts, data.HasCrCard, data.IsActiveMember, data.EstimatedSalary])
    
    # TO DF
    input_data = pd.DataFrame([input_data], columns=columns)

    # Modify Data Types
    X_new = input_data.astype(dtypes)

    # Call pipeline
    processed = pipe.transform(X_new)

    # Model prediction
    y_pred = model.predict(processed)[0]

    case = 'Exited' if y_pred==1 else 'Not Exited'

    return case