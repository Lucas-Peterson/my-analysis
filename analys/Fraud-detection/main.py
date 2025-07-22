import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Download dataset using kagglehub API
print("Downloading dataset from KaggleHub...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
dataset_path = f"{path}/creditcard.csv"
print(f"Path to dataset: {dataset_path}")

# Load dataset
print("Loading dataset...")
data = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")

# EDA: Class distribution
plt.figure(figsize=(7, 5))
sns.countplot(x='Class', data=data, color='steelblue')
plt.title("Class Distribution (0: Normal Transactions, 1: Fraudulent Transactions)", fontsize=16)
plt.xlabel("Class (0: Normal, 1: Fraud)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(ticks=[0, 1], labels=["Normal", "Fraud"], fontsize=12)
plt.grid(True)
plt.show()

# EDA: Transaction amount distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Amount'], bins=50, kde=True, color='blue')
plt.title("Distribution of Transaction Amounts", fontsize=16)
plt.xlabel("Transaction Amount (USD)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xlim([0, 3000])
plt.grid(True)
plt.show()

# Data Preprocessing
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['Time_scaled'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
data = data.drop(['Amount', 'Time'], axis=1)

# Split data into features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print("Logistic Regression trained.")

y_pred_lr = lr.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr))
roc_auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:, 1])

# Random Forest (with class_weight='balanced')
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
print("Random Forest trained.")

y_pred_rf = rf.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))
roc_auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])

# ROC Curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})', color='blue', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})', color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing', color='gray', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.title('ROC Curve Comparison: Logistic Regression, Random Forest', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.show()

# --- Function for predicting new transaction ---
def predict_new_transaction(input_dict):
    """
    input_dict — a dictionary like: {'V1': value, ..., 'V28': value, 'Amount': value, 'Time': value}
    """
    
    amount_scaled = scaler.transform([[input_dict['Amount']]])[0][0]
    time_scaled = scaler.transform([[input_dict['Time']]])[0][0]
    # Build features in the same order as in X_train
    features = []
    for col in X.columns:
        if col == 'Amount_scaled':
            features.append(amount_scaled)
        elif col == 'Time_scaled':
            features.append(time_scaled)
        else:
            features.append(input_dict[col])
    features_df = pd.DataFrame([features], columns=X.columns)

    prob_lr = lr.predict_proba(features_df)[:, 1][0]
    prob_rf = rf.predict_proba(features_df)[:, 1][0]
    print(f"Logistic Regression: fraud probability = {prob_lr*100:.2f}%")
    print(f"Random Forest:      fraud probability = {prob_rf*100:.2f}%")
    return prob_lr, prob_rf

# --- Example of usage ---
example_transaction = {
    'V1': 0.1, 'V2': -0.3, 'V3': 1.2, 'V4': 0.5, 'V5': -0.7, 'V6': 0.3, 'V7': -0.1,
    'V8': 0.2, 'V9': -0.5, 'V10': 1.1, 'V11': 0.4, 'V12': -0.8, 'V13': 0.7, 'V14': -0.2,
    'V15': 0.0, 'V16': 0.6, 'V17': -0.4, 'V18': 1.5, 'V19': -0.3, 'V20': 0.1, 'V21': 0.2,
    'V22': -0.6, 'V23': 0.3, 'V24': -0.7, 'V25': 0.8, 'V26': 0.1, 'V27': -0.2, 'V28': 0.4,
    'Amount': 100.0,    # transaction amount in $
    'Time': 80000       # time
}

predict_new_transaction(example_transaction)
