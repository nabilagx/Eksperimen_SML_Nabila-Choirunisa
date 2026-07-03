import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import os

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "preprocessing",
    "Churn_Modelling_clean.csv"
)

df = pd.read_csv(DATA_PATH)

X = df.drop("Exited", axis=1)
y = df["Exited"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

mlflow.set_experiment("Churn_Modelling")

mlflow.sklearn.autolog()

with mlflow.start_run():

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("="*50)
    print("HASIL EVALUASI")
    print("="*50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
