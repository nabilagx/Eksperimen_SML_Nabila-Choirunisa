import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess(input_path, output_path):

    df = pd.read_csv(input_path)

    df.drop(
        columns=["RowNumber", "CustomerId", "Surname"],
        inplace=True
    )

    gender_encoder = LabelEncoder()
    df["Gender"] = gender_encoder.fit_transform(df["Gender"])

    geo_encoder = LabelEncoder()
    df["Geography"] = geo_encoder.fit_transform(df["Geography"])

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    X_scaled["Exited"] = y.values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X_scaled.to_csv(output_path, index=False)

    print("Preprocessing selesai.")
    print("Dataset tersimpan di:", output_path)


if __name__ == "__main__":

    preprocess(
        "Churn_Modelling.csv",
        "preprocessing/Churn_Modelling_clean.csv"
    )