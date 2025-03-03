import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_results(input_file="risk_data.csv"):
    """
    Plots the Actual vs. Predicted Initial Margin and Value at Risk (VaR) metrics.

    # Purpose:
    - This function visualizes risk metrics over time to analyze market fluctuations.
    - It plots **Historical VaR** and **Parametric VaR**, which are essential for understanding 
      market risk exposure and margin requirements.
    - It helps in **risk assessment and capital allocation**.

    # Why Plot VaR Metrics?
    - Historical VaR (95%) measures risk based on past observed price changes.
    - Parametric VaR (95%) uses statistical models such as the normal distribution to estimate future risk.
    - Comparing both helps determine if historical patterns align with theoretical risk models.

    # Parameters:
    input_file (str): Path to the CSV file containing market and risk data.

    # Returns:
    - None (It displays a plot showing VaR trends over time with improved readability and formatting).
    """
    
    # Load data
    data = pd.read_csv(input_file, parse_dates=['Date'], index_col='Date')

    # Plot Value at Risk (VaR)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Historical_VaR'], label='Historical VaR (95%)', color='purple', linewidth=2)
    plt.plot(data.index, data['Parametric_VaR'], label='Parametric VaR (95%)', color='orange', linestyle='dashed', linewidth=2)
    
    # Highlight areas where Parametric VaR deviates significantly
    plt.fill_between(data.index, data['Historical_VaR'], data['Parametric_VaR'], color='gray', alpha=0.2)
    
    # Labels and Title
    plt.xlabel("Date")
    plt.ylabel("VaR")
    plt.title("Value at Risk (VaR) Analysis")
    plt.legend()
    
    # Improve Date Formatting
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.grid(alpha=0.5)  # Make gridlines lighter for better readability
    

    # Save as high-quality PNG
    plt.savefig("value_at_risk_analysis.png", dpi=300, bbox_inches='tight') 
    # Save as PDF
    plt.savefig("value_at_risk_analysis.pdf", bbox_inches='tight') 
    # Display the plot
    plt.show()  

if __name__ == "__main__":
    plot_results()
