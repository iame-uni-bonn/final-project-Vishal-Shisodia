import yfinance as yf
import pandas as pd

def fetch_and_save_data(symbol="NG=F", start_date="2021-01-01", end_date="2024-01-01", filename="natural_gas_futures.csv"):
    """
    #Fetches historical market data for a given futures contract from Yahoo Finance and saves it as a CSV file.

    Following are the parameters:
    1) symbol (str): The ticker symbol of the futures contract (e.g., "NG=F" for Natural Gas Futures).
    2) start_date (str): The start date for fetching data in "YYYY-MM-DD" format.
    3) end_date (str): The end date for fetching data in "YYYY-MM-DD" format.
    4) filename (str): Name of the CSV file to save the data.

    Returns:
    None as the function does not return any value explicitly. Instead, it performs actions such as fetching data and saving it to a CSV file.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    data.to_csv(filename)
    print(f"Data saved to {filename}")
if __name__ == "__main__":
    fetch_and_save_data()
