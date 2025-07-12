#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Data preparation
df = pd.read_csv("Mock_Telecom_Churn_Data.csv")
df['internet_service'].fillna('unknown', inplace=True)
df['avg_monthly_charge'] = df['total_charges'] / df['tenure'].replace(0, np.nan)
df['is_short_contract'] = df['contract_type'].apply(lambda x: 1 if x == 'month-to-month' else 0)
df = pd.get_dummies(df, columns=['contract_type', 'internet_service'], drop_first=True)
df['churn'] = df['churn'].map({'No': 0, 'Yes': 1})

# Drop
X = df.drop(['customer_id', 'churn'], axis=1)
y = df['churn']

# SMOTE 
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Training and evaluation
for name, model in models.items():
    print(f"\n\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    if hasattr(model, "predict_proba"):
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"ROC AUC Score: {roc_auc:.4f}")
    cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='f1')
    print(f"Cross-validated F1 scores: {cv_scores}")
    print(f"Average F1 score: {cv_scores.mean():.4f}")


# Feature importance
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importances - Random Forest')
plt.tight_layout()
plt.show()


# Prediction function
def preprocess_input(new_client: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    new_client['internet_service'].fillna('unknown', inplace=True)
    new_client['avg_monthly_charge'] = new_client['total_charges'] / new_client['tenure'].replace(0, np.nan)
    new_client['is_short_contract'] = new_client['contract_type'].apply(lambda x: 1 if x == 'month-to-month' else 0)

    new_client = pd.get_dummies(new_client, columns=['contract_type', 'internet_service'], drop_first=True)

    for col in reference_columns:
        if col not in new_client.columns:
            new_client[col] = 0

    new_client = new_client[reference_columns]
    return new_client

# Prediction function
def predict_churn_for_client(model, new_client_raw: pd.DataFrame, reference_columns: list):
    new_client = preprocess_input(new_client_raw.copy(), reference_columns)
    prob = model.predict_proba(new_client)[:, 1][0]
    return f"Customer Churn Probability: {prob:.2%}"

# Example usage
new_client = pd.DataFrame([{
    'tenure': 5,
    'total_charges': 250,
    'contract_type': 'month-to-month',
    'internet_service': 'fiber optic',
}])

reference_columns = list(X.columns)
result = predict_churn_for_client(models["Gradient Boosting"], new_client, reference_columns)
print(result)
