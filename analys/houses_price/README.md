# House Prices Prediction using Random Forest

## Dataset ---> https://www.kaggle.com/datasets/samuelcortinhas/house-price-prediction-seattle


This project predicts house prices using the `RandomForestRegressor` model from the `sklearn` library. It processes the data, removes unnecessary columns, fills missing values, and outputs feature importance to help visualize which factors most influence house prices.

## Features

- Data preprocessing: Handles missing values and removes unnecessary columns.
- House price prediction using the Random Forest algorithm.
- Visualization of feature importance.

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.x
- Required Python packages:
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn matplotlib

```
## Data

train.csv: Training data with house features and prices.

test.csv: Test data for evaluating the model performance.

Both CSV files should contain a price column representing house prices and other feature columns like house size, number of rooms, etc. The columns size_units and lot_size_units will be dropped during preprocessing.
