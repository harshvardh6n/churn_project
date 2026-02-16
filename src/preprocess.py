
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df.replace({'Yes':1,'No':0}, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler
