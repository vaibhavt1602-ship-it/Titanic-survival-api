import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model once when server starts
app = FastAPI()
model = joblib.load('titanic_model.pkl')

# Define what data you expect
class Passenger(BaseModel):
    pclass: int
    sex: int
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked_q: int
    embarked_s: int
    family_size: int
    is_alone: int

@app.get('/')
def home():
    return {"message": "Titanic Survival API is running"}

@app.post('/predict')
def predict(passenger: Passenger):
    # Convert to DataFrame
    data = pd.DataFrame([{
        'Pclass': passenger.pclass,
        'Sex': passenger.sex,
        'Age': passenger.age,
        'SibSp': passenger.sibsp,
        'Parch': passenger.parch,
        'Fare': passenger.fare,
        'Embarked_Q': passenger.embarked_q,
        'Embarked_S': passenger.embarked_s,
        'FamilySize': passenger.family_size,
        'IsAlone': passenger.is_alone
    }])

    # Predict
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "survived": bool(prediction),
        "probability": f"{probability:.2%}"
    }