# Margin Calculation and Risk Analysis for Financial Derivatives

This project aims to develop a Python tool to calculate margin requirements and assess risk exposure for financial derivatives traded on an exchange. The tool is designed to simplify margining processes and provide key risk metrics such as Value at Risk (VaR). 

## Motivation  
Margining plays a crucial role in financial markets by ensuring that traders maintain adequate collateral to cover potential losses. Given my experience as a working student at **XXX**, an energy company in the **Exchange, Clearing, and Marketing Operations** department, I have developed a deeper understanding of margin requirements and risk assessment. This exposure has inspired me to work on a computational tool that simplifies margin calculations and provides insights into risk exposure.

The project aims to bridge the gap between theoretical finance concepts and practical implementation through **Python-based automation**.

## Reproducibility Statement
This project is designed to be **fully reproducible**, ensuring that anyone can replicate the results under the same conditions. It meets the reproducibility standard through:
- Code Availability: All scripts are clearly documented and openly available.
- Data Sources: The project fetches historical price data from `yfinance`, making it accessible for any user.The data is from 2021-01-01 to 2024-01-01.
- Execution Steps: The instructions for running the project are included in this file, with well-commented Python scripts for clarity.
- Version Control: The project is managed using Git, maintaining track of code changes for reproducibility.

## Features
- Data Collection & Cleaning: Fetches market data, processes it, and prepares it for analysis.
- Initial Margin Calculation: Computes initial margin based on historical price data and volatility.
- Variation Margin Tracking: Tracks daily profit/loss and adjusts variation margin dynamically.
- Risk Analysis: Includes key risk metrics like Value at Risk (VaR) to assess potential losses.
- Machine Learning Model: Uses a Random Forest model to predict initial margin requirements.
- Visualization: Plots historical price trends, margin requirements, and risk metrics for better insights.


### Installation Steps

# Clone the repository
git clone <https://github.com/iame-uni-bonn/final-project-Vishal-Shisodia.git>


### Step 1: Fetch Market Data
```sh
python fetch_data.py
```
This script fetches historical market data from yahoo finance for the period 2021-01-01 to 2024-01-01 and saves it as a CSV file.

### Step 2: Clean and Process Data
```sh
python clean_data.py
```
1) Data Loading:
This script processes the financial data from a CSV file (natural_gas_futures.csv). 
The dataset is loaded using pandas.read_csv().
The script skips the first two rows while reading the CSV file, which ensures only relevant financial data is included.
2) Column Renaming:
To standardize the dataset, column names were renamed for clarity: ["Close", "High", "Low", "Open", "Volume"]
This ensures uniformity across different datasets and simplifies analysis.
3) Handling Missing Values:
Missing values are dropped using data.dropna(inplace=True).
This prevents computational issues in later steps, ensuring clean and reliable data.
4) Feature Engineering:
Daily Returns Calculation Computes daily percentage change in closing price and helps in risk assessment and volatility analysis.
Rolling Volatility Calculation, A 10-day rolling window standard deviation of returns is used it represents historical price fluctuations, which is crucial for margin calculations.

### Step 3: Compute Risk Metrics
```sh
python risk_analysis.py
```
This script calculates Initial Margin, Variation Margin, and Value at Risk (VaR).

### Step 4: Train Machine Learning Model
```sh
python train_model.py
```
This script trains a Random Forest model to predict Initial Margin based on historical risk metrics.

### Step 5: Visualize Risk Metrics
```sh
python visualize_results.py
```
This script generates plots for VaR and other risk metrics.

## Project Structure
```
/margin-requirements-project
│── data/                      # Directory to store raw and processed data
│── fetch_data.py              # Fetches financial data
│── clean_data.py              # Cleans and preprocesses data
│── risk_analysis.py           # Computes risk metrics
│── train_model.py             # Trains machine learning model
│── visualize_results.py       # Generates risk analysis plots
│── README.md                  # Documentation for the project
│── report.pdf                 # Final project report (both the pdf and LaTeX script)

```

## Libraries Used
All the packages are already installed in the course environment "epp".

  - python=3.11.0
  - pandas=2.2.3
  - numpy=2.0.2
  - matplotlib=3.9.2
  - scikit-learn=1.5.2
  - pip2=4.2
  - yfinance==0.2.54

### Data Processing & Computation
- `pandas` – Data manipulation, handling CSV files, and computing rolling statistics.
- `numpy` – Mathematical operations, including volatility and profit/loss calculations.

### Financial Data Retrieval
- `yfinance` – Fetches historical financial data (e.g., natural gas futures prices).

### Statistical & Risk Analysis
- `scikit-learn` – Machine learning for risk prediction.

### Visualization 
- `matplotlib` – Generates plots for price trends, margin requirements, and risk metrics.

## Methodology
1. Fetch Data – Retrieve historical market prices via `yfinance`.
2. Data Cleaning – Process and prepare raw data for analysis.
3. Initial Margin Calculation – Compute using historical price volatility.
4. Variation Margin Calculation – Track daily profit/loss and adjust accordingly.
5. Risk Metrics Computation – Calculate VaR using historical and parametric methods.
6. Machine Learning Model – Train a Random Forest model to predict Initial Margin.
7. Data Visualization – Graphically represent results using `matplotlib`.

## Results & Insights
1) Model Performance:

Mean Squared Error (MSE): 0.000782 (Lower values indicate better predictions.)
R-Squared Score (R²): 0.982 (The model explains 98.2% of variance in Initial Margin.)

2) Feature Importance Analysis:

Parametric VaR is the most influential factor in Initial Margin prediction (94.3% importance).
Volatility also contributes significantly (4.5% importance).
Historical VaR and short-term variations in margin play minor roles.

3) Value at Risk (VaR) Trends:

Historical VaR and Parametric VaR exhibit different behaviors, especially during volatile market periods.
The difference between them highlights potential underestimations or overestimations of risk in traditional models.

4) Margin Fluctuations:

High volatility increases margin requirements, ensuring sufficient collateral coverage.
Variation Margin effectively captures daily profit/loss dynamics.

## Limitations & Future Work
- Assumes historical volatility accurately predicts future margins.
- Could integrate deep learning models for improved predictions.
- Future versions may include more complex risk models such as Expected Shortfall (ES).

## References

### Books & Industry Guidelines
- Hull, J. (2017). *Options, Futures, and Other Derivatives*. Pearson.
- Basel Committee on Banking Supervision. (2013). *Margin Requirements for Non-Centrally Cleared Derivatives*.

### Libraries & Tools
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [matplotlib Documentation](https://matplotlib.org/stable/contents.html)



