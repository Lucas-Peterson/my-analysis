# Stock Price Analysis and Prediction using Polynomial Regression

This project is designed to analyze historical stock price data for several major companies and use polynomial regression to predict future stock prices. The data is fetched using Yahoo Finance (`yfinance`), and the predictions are made with the help of polynomial regression using the `scikit-learn` library.

## Features
- **Fetch Historical Data**: Automatically fetches stock price data for selected companies using Yahoo Finance.
- **Data Visualization**: Visualizes historical closing prices for each stock.
- **Polynomial Regression**: Uses polynomial regression to model and predict future stock prices.
- **Customizable Forecast**: The prediction horizon and polynomial degree can be easily modified.

## Stock Symbols
The following stock symbols are currently used in the project:
- Apple (AAPL)
- Microsoft (MSFT)
- Nvidia (NVDA)

You can add more stock symbols by modifying the `symbols` list in the code.

## Requirements

The project uses the following Python packages:
- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the required packages by running the following command:

```bash
pip install -r requirements.txt
```


## Installation

**Clone the repository**:
```bash
git clone https://github.com/your-username/stock-analysis.git
```
**Navigate into the project directory**:
```bash
cd stock-analysis
```
**Install the necessary dependencies**:
```bash
pip install -r requirements.txt
```
**Run the project**:
```bash
python main.py
```

## Structure

Usage
Data Fetching
The script will automatically fetch the stock data for the specified symbols and save it in the stock_data directory in CSV format.

Visualization
The project visualizes the historical closing prices of the stocks using line plots. These plots are saved in the stock_data directory as PNG files.

Polynomial Regression
The project uses polynomial regression to predict future stock prices. The degree of the polynomial and the number of future days for which predictions are made can be modified in the following line:

```bash
polynomial_regression_prediction(df_stocks, symbol, degree=3, future_days=30)
```

degree=3: The degree of the polynomial used for regression (default is 3).
future_days=30: The number of future days for which predictions will be made.


## Output
Stock Close Prices Plot: A comparison plot showing the historical closing prices for the selected stocks.
Polynomial Regression Plot (Test Set): A plot showing actual vs predicted closing prices for the test set.
Future Price Predictions: A plot showing the forecast for the next specified number of days based on the trained polynomial regression model.


Structure
```bash
├── stock_data                 # Directory for storing stock data and plots
├── main.py                    # Main Python script
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```


### Customization
Changing the stock symbols: Edit the symbols list in the main.py file to analyze different stocks.
Changing the date range: Modify the start_date and end_date variables to fetch data for a different time period.
Changing the polynomial degree or forecast length: Modify the degree and future_days parameters in the polynomial_regression_prediction function to adjust the complexity of the model and the forecast horizon.

### Example Output

2023-10-06 12:05:31,742 - INFO - Fetching new stock data from Yahoo Finance...
2023-10-06 12:05:33,905 - INFO - Stock data saved to CSV.
2023-10-06 12:05:33,905 - INFO - First few rows of stock data:
               Close                                             
               AAPL        MSFT        NVDA
Date                                          
2020-01-02  75.087502  160.619995  59.810001
2020-01-03  74.287498  158.619995  59.140000
2020-01-06  74.949997  159.029999  60.250000
...
2023-10-06 12:05:46,324 - INFO - Data successfully loaded.
