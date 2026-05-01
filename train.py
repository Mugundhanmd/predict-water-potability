import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

def train_and_save():
    # Load dataset
    df = pd.read_csv('data/water_potability.csv')
    
    # Preprocess missing values
    for col in df.columns:
        if col != 'Potability':
            df[col] = df[col].fillna(df[col].mean())
            
    # Split features and target
    X = df.drop(columns=['Potability'])
    y = df['Potability']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    accuracy = model.score(scaler.transform(X_test), y_test)
    print(f"Model trained successfully! Accuracy on test set: {accuracy * 100:.2f}%")
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    with open('models/xgboost.sav', 'wb') as f:
        pickle.dump(model, f)
    with open('models/scaler.sav', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved successfully to 'models/' directory.")

if __name__ == '__main__':
    train_and_save()
