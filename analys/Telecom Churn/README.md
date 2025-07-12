# Telecom Churn Prediction

This project contains a complete example of building a churn prediction model using machine learning techniques.

## Overview

The notebook includes the following steps:
- Loading and preprocessing customer data
- Balancing the dataset using SMOTE
- Training and evaluating three models:
  - Random Forest
  - Logistic Regression
  - Gradient Boosting
- Visualizing feature importance
- Making predictions for a new customer

## Dataset

The dataset used in this project is randomly generated for demonstration purposes. It simulates a telecom customer base and is named `Mock_Telecom_Churn_Data.csv`. The file is included in the project directory.

## Models

Each model is trained using the same features and evaluated with:
- Confusion matrix
- Classification report
- ROC AUC score
- Cross-validated F1 scores

## Output

The results summary and a feature importance plot from the Random Forest model are included at the end of the notebook.

## Requirements

To run this project, make sure the following Python libraries are installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

You can install them using pip:

```
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## Files

- `telecom_churn_analysis.ipynb`: Main analysis notebook
- `Mock_Telecom_Churn_Data.csv`: Randomly generated dataset
- `feature_importance_rf.png`: Feature importance chart
- `telecom_churn_results_only.ipynb`: Notebook with model evaluation summary only