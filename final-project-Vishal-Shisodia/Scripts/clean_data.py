import pandas as pd
import numpy as np
 # Here we load the CSV data, process it and add the computed features and save the clean data.

def preprocess_data(input_file="./final-project-Vishal-Shisodia/Data/natural_gas_futures.csv", output_file="./final-project-Vishal-Shisodia/Data/processed_data.csv"):


    """

   #Following are the Steps:
    1. Reads data from a CSV file.
    2. Renames columns for consistency.
    3. Drops missing values.
    4. Computes daily returns and volatility.
    5. Saves the processed data.

    Parameters:
    input_file (str): Name of the input CSV file.
    output_file (str): Name of the processed output CSV file.

    Returns:
    None
    """
    # Load data
    data = pd.read_csv(input_file, skiprows=2, parse_dates=['Date'], index_col='Date')

    # Rename columns for clarity
    data.columns = ["Close", "High", "Low", "Open", "Volume"]

    # Drop missing values
    data.dropna(inplace=True)

    # Compute daily returns
    data['Returns'] = data['Close'].pct_change()

    # Compute rolling volatility (10-day standard deviation of returns)
    data['Volatility'] = data['Returns'].rolling(10).std()

    # Drop NaN values after computations
    data.dropna(inplace=True)

    # Save processed data
    data.to_csv(output_file)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data()
