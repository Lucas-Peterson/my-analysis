import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Models and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# For handling imbalanced data
from imblearn.over_sampling import SMOTE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
logging.info("Loading dataset...")
data = pd.read_csv("creditcard.csv")
logging.info("Dataset loaded successfully.")

# Exploratory Data Analysis (EDA)
logging.info("Starting EDA...")
plt.figure(figsize=(7, 5))
sns.countplot(x='Class', data=data, color='steelblue')
plt.title("Class Distribution (0: Normal Transactions, 1: Fraudulent Transactions)", fontsize=16)
plt.xlabel("Class (0: Normal, 1: Fraud)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(ticks=[0, 1], labels=["Normal", "Fraud"], fontsize=12)
plt.grid(True)
plt.show()

logging.info("EDA completed successfully.")

# Visualize distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(data['Amount'], bins=50, kde=True, color='blue')
plt.title("Distribution of Transaction Amounts", fontsize=16)
plt.xlabel("Transaction Amount (USD)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xlim([0, 3000])  # Limiting to 3000 for better visibility of majority transactions
plt.grid(True)
plt.show()

logging.info("Transaction amount distribution plotted.")

# Data Preprocessing
logging.info("Starting data preprocessing...")
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['Time_scaled'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Drop the original 'Amount' and 'Time' columns
data = data.drop(['Amount', 'Time'], axis=1)

# Split data into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to balance the classes
logging.info("Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
logging.info("SMOTE applied successfully.")

# Model 1: Logistic Regression
logging.info("Training Logistic Regression...")
lr = LogisticRegression()
lr.fit(X_train_res, y_train_res)
logging.info("Logistic Regression trained successfully.")

# Predictions for Logistic Regression
y_pred_lr = lr.predict(X_test)

# Model evaluation for Logistic Regression
logging.info("Evaluating Logistic Regression...")
print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr))

# ROC AUC score for Logistic Regression
roc_auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
logging.info(f"ROC AUC for Logistic Regression: {roc_auc_lr:.2f}")

# ROC curve for Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:, 1])

# Model 2: Decision Tree
logging.info("Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_res, y_train_res)
logging.info("Decision Tree trained successfully.")

# Predictions for Decision Tree
y_pred_dt = dt.predict(X_test)

# Model evaluation for Decision Tree
logging.info("Evaluating Decision Tree...")
print("Decision Tree:")
print(classification_report(y_test, y_pred_dt))

# ROC AUC score for Decision Tree
try:
    roc_auc_dt = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
    logging.info(f"ROC AUC for Decision Tree: {roc_auc_dt:.2f}")
except AttributeError as e:
    logging.error(f"Decision Tree does not support probability prediction: {e}")

# ROC curve for Decision Tree
try:
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt.predict_proba(X_test)[:, 1])
except AttributeError as e:
    logging.error(f"Decision Tree does not support probability prediction: {e}")

# Model 3: Random Forest
logging.info("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
logging.info("Random Forest trained successfully.")

# Predictions for Random Forest
y_pred_rf = rf.predict(X_test)

# Model evaluation for Random Forest
logging.info("Evaluating Random Forest...")
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))

# ROC AUC score for Random Forest
roc_auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
logging.info(f"ROC AUC for Random Forest: {roc_auc_rf:.2f}")

# ROC curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])

# Visualize ROC curves for all models
logging.info("Plotting ROC curves for all models...")
plt.figure(figsize=(10, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})', color='blue', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})', color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing', color='gray', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.title('ROC Curve Comparison: Logistic Regression, Decision Tree, Random Forest', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.show()

logging.info("ROC curves plotted successfully.")
