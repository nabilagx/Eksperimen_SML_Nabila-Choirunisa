import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing(input_path, output_dir):
    print("Memulai otomatisasi preprocessing data...")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} tidak ditemukan!")
    df = pd.read_csv(input_path)

    kolom_tidak_penting = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=kolom_tidak_penting, errors='ignore')

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    if 'Geography' in df.columns:
        df['Geography'] = le_geo.fit_transform(df['Geography'])
    if 'Gender' in df.columns:
        df['Gender'] = le_gender.fit_transform(df['Gender'])

    scaler = StandardScaler()
    kolom_numerik = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[kolom_numerik] = scaler.fit_transform(df[kolom_numerik])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'Churn_Modelling_clean.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Otomatisasi sukses! Data siap latih disimpan di: {output_path}")

if __name__ == "__main__":
    run_preprocessing('Churn_Modelling.csv', 'preprocessing')