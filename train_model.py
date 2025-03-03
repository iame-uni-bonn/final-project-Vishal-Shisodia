import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Trains and evaluates a Linear Regression model to predict Initial Margin for futures trading.

def train_model(input_file="risk_data.csv"):
    """
    # The Purpose of doing these are as follows:

    - The Initial Margin is a required deposit for trading futures contracts.
    - Here we use **historical data** to predict future Initial Margin requirements.
    - A reliable model helps in *risk management* and *capital allocation*.

    # Methodology 

    1. **Load Data**:
       - Reads the processed risk data from `risk_data.csv`.
       - Selects relevant features for margin prediction.
    
    2. **Feature Selection**:
       - Uses **Close Price** and **Volatility** as predictor variables.
       - Sets **Initial Margin** as the target variable.

    3. **Data Preprocessing**:
       - Splits data into **Training (80%)** and **Test (20%)** sets.
       - Applies **StandardScaler** to normalize feature values.

    4. **Model Training**:
       - Trains a **Linear Regression model** on the training data.

    5. **Model Evaluation**:
       - It computes **Mean Squared Error (MSE)** which measures the average squared error.
       - It computes **R² Score** which indicates how well the model explains variance in Initial Margin.
       - It prints the evaluation metrics.


    # Parameters
    input_file(str): Path to the input CSV file containing risk data.

    # Returns
    - None (It displays model performance metrics).
    
    """
    
    # Load data
    data = pd.read_csv(input_file, parse_dates=['Date'], index_col='Date')

    # Define features and target
    X = data[['Close', 'Volatility']]
    y = data['Initial_Margin']

    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Standardize features (it is important for consistent model performance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear Regression model
    model = LinearRegression()
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
