import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns

# Load data
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=wine_data.target_names, yticklabels=wine_data.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine_data.target_names))

# Feature Importance
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]

print("\nFeature Importances:")
for i in range(X.shape[1]):
    print(f"{wine_data.feature_names[indices[i]]}: {feature_importances[indices[i]]:.4f}")

# Plot Feature Importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), [wine_data.feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Cross-validation score
cross_val = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {cross_val.mean():.2f} ± {cross_val.std():.2f}")

# ROC and AUC (for multiclass classification)
y_prob = model.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(len(wine_data.target_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc[i] = roc_auc_score(y_test == i, y_prob[:, i])

# Plot ROC curves
plt.figure(figsize=(10, 6))
for i in range(len(wine_data.target_names)):
    plt.plot(fpr[i], tpr[i], label=f"Class {wine_data.target_names[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Wine Classification')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
