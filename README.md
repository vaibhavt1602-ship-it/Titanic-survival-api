# Titanic Survival Predictor API

Predicts whether a Titanic passenger would have survived
based on their details. Built as a REST API so anyone can use it.

## Results
- 82% accuracy on test set
- 0.744 public score on Kaggle leaderboard

## Tech Stack
- Python, FastAPI, scikit-learn, pandas, joblib

## Project Structure
titanic-survival-api/
├── pipeline.py        # data cleaning, feature engineering, model training
├── main.py            # FastAPI server
├── titanic_model.pkl  # saved Random Forest model
├── requirements.txt   # dependencies
└── README.md

## How to Run
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Then open http://127.0.0.1:8000/docs to test the API.

## Example Prediction
POST /predict
```json
{
  "pclass": 3,
  "sex": 1,
  "age": 22,
  "sibsp": 0,
  "parch": 0,
  "fare": 7.25,
  "embarked_q": 0,
  "embarked_s": 1,
  "family_size": 1,
  "is_alone": 1
}
```
Response:
```json
{
  "survived": false,
  "probability": "1.00%"
}
```

## What I Learned
- Full ML pipeline on real messy data
- Feature engineering (FamilySize, IsAlone, title extraction)
- Serving a model via REST API with FastAPI
- Input validation with Pydantic
