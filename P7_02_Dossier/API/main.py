# This is a sample Python script.

import pickle

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# On crée notre instance FastApi puis on définit l'objet enstiment
app = FastAPI()


class ScroringCredit(BaseModel):
    credit_bank_decision: str
    credit_bank_proba: float
    message: str


# On crée notre pipeline
model_load = pickle.load(open('banking_model.md', 'rb'))
data = pd.read_csv('x_train_fastapi.csv', index_col=0)
best_parameters = {'colsample_by_tree': 0.6000000000000001,
                   'learning_rate': 0.026478707430398492,
                   'max_depth': 28.0,
                   'n_estimators': 1000.0,
                   'num_leaves': 4.0,
                   'reg_alpha': 0.8,
                   'reg_lambda': 0.7000000000000001,
                   'solvability_threshold': 0.25,
                   'subsample': 0.8}


# Nos différents endpoints

@app.get('/')
def get_root():
    return {'message': 'Welcome to the bank credit analysis API'}


@app.get('/predict/{id_client}', response_model=ScroringCredit)
async def predict(id_client: int):

    prediction_result = model_load.predict_proba(pd.DataFrame(data.loc[id_client]).transpose())[:, 1]
    prediction_bool = np.array((prediction_result > best_parameters['solvability_threshold']) > 0) * 1
    if prediction_bool == 0:
        prediction_decision = 'loan approved'
    else:
        prediction_decision = 'loan refused'
    return ScroringCredit(credit_bank_decision=prediction_decision, credit_bank_proba=prediction_result)


if __name__ == "__main__":
    uvicorn.run("main:app")
