import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the datasets
train_data = pd.read_csv('/Users/V/PycharmProjects/wine/train.csv')
test_data = pd.read_csv('/Users/V/PycharmProjects/wine/test.csv')

# Remove unnecessary columns and fill missing values
columns_to_drop = ['size_units', 'lot_size_units']
train_data = train_data.drop(columns=columns_to_drop)
test_data = test_data.drop(columns=columns_to_drop)

# Handle missing values using SimpleImputer with median strategy
imputer = SimpleImputer(strategy='median')
X_train = train_data.drop('price', axis=1)
y_train = train_data['price']
X_train = imputer.fit_transform(X_train)

X_test = test_data.drop('price', axis=1)
y_test = test_data['price']
X_test = imputer.transform(X_test)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions using Random Forest
y_pred_rf = rf_model.predict(X_test)

# Calculate Mean Absolute Error and Mean Squared Error
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Calculate RAE (Relative Absolute Error)
numerator = sum(abs(y_test - y_pred_rf))
denominator = sum(abs(y_test - y_test.mean()))
rae_rf = numerator / denominator

# Print evaluation metrics
print(f"Random Forest - MAE: {mae_rf}, MSE: {mse_rf}, RAE: {rae_rf}")

# Feature importance
importances = rf_model.feature_importances_
sorted_indices = importances.argsort()

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(train_data.columns.drop('price')[sorted_indices], importances[sorted_indices], color='blue')
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.show()
