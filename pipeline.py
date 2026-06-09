import pandas as pd
import joblib   

df = pd.read_csv('Titanic-Dataset.csv')


#  Drop useless columns
df = df.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)

#  — Fix missing values
df['Age'] = df['Age'].fillna(df['Age'].median())      # median, not mean (why? outliers)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#  Encode Sex (text → number)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

#  Encode Embarked (3 categories → columns)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Engineer 2 new features (your own thinking)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)




df['Embarked_Q'] = df['Embarked_Q'].astype(int)
df['Embarked_S'] = df['Embarked_S'].astype(int)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split X and y
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try 2 models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
print("Training columns:", X_train.columns.tolist())
print("Number of features:", len(X_train.columns))

# Evaluate
print("=== Logistic Regression ===")
print(classification_report(y_test, lr.predict(X_test)))

print("=== Random Forest ===")
print(classification_report(y_test, rf.predict(X_test)))

joblib.dump(rf, 'titanic_model.pkl')
print("Model saved → titanic_model.pkl")


# Test saved model
import numpy as np

loaded = joblib.load('titanic_model.pkl')

# 3rd class male, 22yrs, travelling alone
passenger = [[3, 0, 22.0, 0, 0, 7.25, 0, 1, 1, 1]]
pred = loaded.predict(passenger)
prob = loaded.predict_proba(passenger)[0][1]

# print(f"Survived: {bool(pred[0])}")
# print(f"Survival probability: {prob:.2%}")

# ── Kaggle Submission ───────────────────────────────────────────
test_df = pd.read_csv('test.csv')

# Save passenger IDs for submission file
passenger_ids = test_df['PassengerId']


test_df = test_df.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)
test_df['Embarked_Q'] = test_df['Embarked_Q'].astype(int)
test_df['Embarked_S'] = test_df['Embarked_S'].astype(int)

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)

# Predict
predictions = loaded.predict(test_df)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
print("submission.csv created")
print(submission.head())


