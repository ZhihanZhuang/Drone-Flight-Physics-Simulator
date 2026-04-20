import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_rf_model(df):
    """
    Trains a Random Forest Regression model to predict flight time based on hardware specs,
    allowing bypassing of the computational physics engine for rapid estimation.
    """
    os.makedirs("ml_model/saved", exist_ok=True)
    feasible_df = df[df['feasible'] == 1].copy()

    if feasible_df.empty:
        return None

    # Features and Target
    X = feasible_df[['mass', 'motor_k', 'battery']]
    y = feasible_df['flight_time']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Random Forest Model Trained Successfully")
    print(f"   -> R² Score (Accuracy): {r2:.3f}")
    print(f"   -> Mean Squared Error:  {mse:.1f}")

    # Save model for future rapid predictions
    joblib.dump(model, "ml_model/saved/flight_time_predictor.pkl")
    return model