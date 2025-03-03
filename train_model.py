import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Trains and evaluates a Random Forest model to predict Initial Margin for futures trading.

def train_model(input_file="risk_data.csv"):
    """
    # Purpose of This Model:

    - **Initial Margin** is a required deposit for trading futures contracts.
    - This model uses **historical data** to predict future Initial Margin requirements.
    - A reliable prediction model helps in *risk management* and *capital allocation*.

    # Why Use Random Forest Instead of Linear Regression?

    - Non-linearity: Initial Margin is influenced by complex, non-linear factors such as volatility and Value at Risk (VaR), which Linear Regression cannot capture well.
    - Better Accuracy: Random Forest averages multiple Decision Trees, reducing variance and improving prediction accuracy.
    - Feature Importance: It helps identify which financial indicators impact Initial Margin the most.
    - Robustness: Handles outliers and missing data better than Linear Regression.

    # Methodology 

    1. **Load Data**:
       - Reads the processed risk data from `risk_data.csv`.
       - Selects relevant features for margin prediction.
    
    2. **Feature Engineering**:
       - Uses **Close Price**, **Volatility**, **5-day rolling averages**, and **VaR metrics**.
       - Computes **relative change in Variation Margin** to account for recent fluctuations.
       
    3. **Data Preprocessing**:
       - Splits data into **Training (80%)** and **Test (20%)** sets.
       - Applies **StandardScaler** to normalize feature values for better model performance.

    4. **Model Training**:
       - Trains a **Random Forest Regressor** with 100 decision trees on the training data.

    5. **Model Evaluation**:
       - It computes **Mean Squared Error (MSE)** which measures the average squared difference between actual and predicted values.
       - It computes **R² Score** which indicates how well the model explains the variance in Initial Margin.
       - It Prints the evaluation metrics.

    # Parameters
    input_file (str): Path to the input CSV file containing risk data.

    # Returns
    - None (It displays model performance metrics).
    """
    
    # Load data
    data = pd.read_csv(input_file, parse_dates=['Date'], index_col='Date')

    # Feature Engineering
    data["Volatility_5d_avg"] = data["Volatility"].rolling(window=5).mean()
    data["Var_Margin_5d_avg"] = data["Variation_Margin"].rolling(window=5).mean()
    data["Var_Margin_Change"] = data["Variation_Margin"].pct_change()
    
    # Drop NaN values created by rolling calculations
    data = data.dropna()

    # Define features and target
    features = ["Close", "Volatility", "Volatility_5d_avg", "Var_Margin_5d_avg", "Var_Margin_Change", "Historical_VaR", "Parametric_VaR"]
    target = "Initial_Margin"

    X = data[features]
    y = data[target]

    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print model performance
    print(f"Mean Squared Error: {mse}")
    print(f"R-Squared Score: {r2}")

if __name__ == "__main__":
    train_model()
