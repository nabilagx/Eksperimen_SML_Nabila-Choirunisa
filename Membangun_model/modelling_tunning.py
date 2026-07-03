import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import dagshub

from mlflow.models import infer_signature
from sklearn.metrics import RocCurveDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)


BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Churn_Modelling_clean.csv"
)

df = pd.read_csv(DATA_PATH)

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
dagshub.init(
    repo_owner="nabilagx",
    repo_name="SMSML_Nabila-Choirunisa",
    mlflow=True
)

mlflow.set_experiment("Churn_Modelling_Tuning")


param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)


with mlflow.start_run():

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    signature = infer_signature(
        X_train,
        best_model.predict(X_train)
    )

    input_example = X_train.iloc[:5]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)


    mlflow.log_param("best_n_estimators", grid.best_params_["n_estimators"])
    mlflow.log_param("best_max_depth", grid.best_params_["max_depth"])
    mlflow.log_param("best_min_samples_split", grid.best_params_["min_samples_split"])

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)


    mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model",
    signature=signature,
    input_example=input_example
    )


    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.savefig("confusion_matrix.png")

    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    RocCurveDisplay.from_estimator(best_model, X_test, y_test)

    plt.savefig("roc_curve.png")

    plt.close()

    mlflow.log_artifact("roc_curve.png")


    report = classification_report(y_test, y_pred)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values()

    plt.figure(figsize=(8,6))

    importance.plot(kind="barh")

    plt.tight_layout()

    plt.savefig("feature_importance.png")

    plt.close()

    mlflow.log_artifact("feature_importance.png")


    requirements_path = os.path.join(BASE_DIR, "..", "requirements.txt")

    if os.path.exists(requirements_path):
        mlflow.log_artifact(requirements_path)


print("="*50)
print("BEST PARAMETER")
print(grid.best_params_)
print("="*50)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

print("\nTraining selesai.")
print("Silakan buka MLflow.")