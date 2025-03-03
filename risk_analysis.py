import pandas as pd
import numpy as np

def compute_risk_metrics(input_file="processed_data.csv", output_file="risk_data.csv"):
    """
    Computes financial risk metrics for Natural Gas Futures, including Initial Margin, Variation Margin, 
    and Value at Risk (VaR), to assess market exposure and risk management.

    === Risk Metrics Computed ===

    1. **Initial Margin**:
       - Represents the required capital to maintain a futures position.
       - **Formula**:
         Initial Margin = Close Price * Volatility * sqrt(10) * Risk Factor
       - Here, "Volatility" is the rolling standard deviation of **daily returns** over 10 days.
       - A "Risk Factor" of **3** ensures sufficient margin coverage.

    2. **Variation Margin**:
       - Represents the **daily profit or loss** due to price changes.
       - **Formula**:
         Variation Margin = (Today's Close Price - Yesterday's Close Price) * Contract Size
       - Contract size is **10,000** units for Natural Gas Futures.

    3. **Value at Risk (VaR)**:
       - Estimates the potential **worst-case loss** over a given period at a **95% confidence level**.
       - **Two VaR Methods Used**:
         - **Historical VaR (Non-Parametric Approach)**:
           - Uses the **5th percentile of past 252 daily returns** to estimate losses.
           - **Formula**:
             Historical VaR (95%) = 5th Percentile of Returns * Close Price
         - **Parametric VaR (Variance-Covariance Method)**:
           - Assumes normally distributed returns and uses a **Z-score**.
           - At **95% confidence**, the **Z-score is 1.645**, meaning we expect 95% of the losses **not** to exceed this threshold.
           - **Formula**:
             Parametric VaR (95%) = -1.645 * Volatility * Close Price
             
    Parameters:
    - `input_file` (str): CSV file containing preprocessed market data.
    - `output_file` (str): Name of the output CSV file to save computed risk metrics.

    Returns:
    - None (Saves the results in `output_file`).
    """

    # Load data
    data = pd.read_csv(input_file, parse_dates=['Date'], index_col='Date')

    # Compute Initial Margin
    risk_factor = 3
    data['Initial_Margin'] = data['Close'] * data['Volatility'] * np.sqrt(10) * risk_factor

    # Compute Variation Margin
    contract_size = 10000
    data['Variation_Margin'] = data['Close'].diff() * contract_size

    # Compute Value at Risk (VaR)
    confidence_level = 0.95
    z_score = 1.645  # 95% confidence level

    # Historical VaR (5th percentile of past 252 daily returns)
    data['Historical_VaR'] = data['Returns'].rolling(252).quantile(1 - confidence_level) * data['Close']

    # Parametric VaR (Variance-Covariance Method)
    data['Parametric_VaR'] = -z_score * data['Volatility'] * data['Close']

    # Drop NaN values
    data.dropna(inplace=True)

    # Save the computed risk metrics
    data.to_csv(output_file)
    print(f"Risk data saved to {output_file}")

if __name__ == "__main__":
    compute_risk_metrics()
