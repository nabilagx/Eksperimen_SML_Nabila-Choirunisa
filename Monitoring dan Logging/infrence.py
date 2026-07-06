import pandas as pd
import requests
import time

URL = "http://127.0.0.1:5000/invocations"

df = pd.read_csv("preprocessing/Churn_Modelling_clean.csv")

sample = df.drop("Exited", axis=1).iloc[[0]]

payload = {
    "dataframe_split": {
        "columns": sample.columns.tolist(),
        "data": sample.values.tolist()
    }
}

print("Sending inference request...")

while True:

    try:

        r = requests.post(
            URL,
            json=payload,
            headers={"Content-Type":"application/json"},
            timeout=10
        )

        print(
            f"Status : {r.status_code}"
        )

        print(
            r.text
        )

    except Exception as e:

        print(e)

    time.sleep(3)