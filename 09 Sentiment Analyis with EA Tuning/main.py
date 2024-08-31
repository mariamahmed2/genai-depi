from utils import cleaning, text_lemma, text_vec, predect_class
from fastapi import FastAPI


app = FastAPI()


map_label = {

    0: "Negative",
    1: "Positive",
    2: "Neutral"

}

@app.post('/predict')
async def tweet_class(data: str):
    # cleaning 
    cleaned = cleaning(text=data)

    # lemma 
    cleaned_lemma = text_lemma(text=cleaned)

    # vec 
    x_process = text_vec(text=cleaned_lemma)

    # Model 
    y_predict = predect_class(x_process=x_process)

    # map
    final_prediction =  map_label.get(y_predict)

    return (f'predidction is {final_prediction}')

