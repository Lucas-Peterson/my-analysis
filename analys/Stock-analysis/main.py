import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Stock symbols
symbols = ['AAPL', 'MSFT', 'NVDA']

# Create directory for storing data if it doesn't exist
data_dir = Path('stock_data')
data_dir.mkdir(exist_ok=True)


# Step 1: Data Collection
def fetch_stock_data(symbols, start, end, force_update=False):
    csv_path = data_dir / 'stock_data_yf.csv'

    if not csv_path.exists() or force_update:
        logging.info("Fetching new stock data from Yahoo Finance...")
        stock_data = yf.download(symbols, start=start, end=end)
        stock_data.to_csv(csv_path)
        logging.info("Stock data saved to CSV.")
    else:
        logging.info("Stock data already exists, loading from CSV.")

    return pd.read_csv(csv_path, header=[0, 1], index_col=0)


# Set time period for stock data
start_date = '2020-01-01'
end_date = '2024-01-01'

# Fetch data
df_stocks = fetch_stock_data(symbols, start=start_date, end=end_date)
df_stocks.index = pd.to_datetime(df_stocks.index)


# Step 2: Visualization of stock closing prices
def plot_close_prices(df, symbols, save_path=None):
    plt.figure(figsize=(14, 7))

    for symbol in symbols:
        plt.plot(df.index, df['Close'][symbol], label=f'{symbol} Close Price')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Close Prices')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Plot saved to {save_path}")
    plt.show()


# Plot stock close prices
plot_path = data_dir / 'stocks_comparison.png'
plot_close_prices(df_stocks, symbols, save_path=plot_path)


# Step 3: Polynomial Regression for predictions
def polynomial_regression_prediction(df, symbol, degree=2, future_days=30):
    # Extract the data for a specific stock symbol
    stock_close = df['Close'][symbol].dropna()

    # Create a feature matrix based on the index (time in days)
    X = np.arange(len(stock_close)).reshape(-1, 1)  # Time as a feature
    y = stock_close.values  # Close price

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with polynomial features and linear regression
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Plot the actual vs predicted prices for the test set
    plt.figure(figsize=(10, 5))
    plt.plot(X_test, y_test, 'b.', label='Actual Close Price')
    plt.plot(X_test, y_pred, 'r-', label='Predicted Close Price (Test)')
    plt.title(f'Polynomial Regression (Degree {degree}) for {symbol} Close Prices')
    plt.xlabel('Days since start')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Future prediction for the next `future_days`
    future_X = np.arange(len(stock_close), len(stock_close) + future_days).reshape(-1, 1)
    future_pred = model.predict(future_X)

    # Plot future predictions
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(stock_close)), stock_close.values, 'b-', label='Historical Close Price')
    plt.plot(future_X, future_pred, 'g--', label=f'Future {future_days} Days Prediction')
    plt.title(f'Future Prediction (Next {future_days} Days) for {symbol}')
    plt.xlabel('Days since start')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


# Predict and plot for each stock using polynomial regression
for symbol in symbols:
    polynomial_regression_prediction(df_stocks, symbol, degree=5, future_days=30)
