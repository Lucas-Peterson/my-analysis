# Credit Card Fraud Detection with Machine Learning

## Overview

This project involves building machine learning models to detect fraudulent credit card transactions. The dataset used for this analysis contains transactions labeled as normal or fraudulent, and the aim is to predict whether a given transaction is fraudulent or not. I use the dataset "Credit Card Fraud Detection Dataset" from Université Libre de Bruxelles (ULB)

Three different models are used in this project:
1. Logistic Regression
2. Decision Tree
3. Random Forest

The models are trained, evaluated, and compared using various performance metrics, including the ROC-AUC score and classification reports.

## Project Structure

1. **Data Loading**: The dataset is loaded from a CSV file named `creditcard.csv`.
2. **Exploratory Data Analysis (EDA)**: 
   - Visualization of the class distribution (normal vs. fraudulent transactions).
   - Visualization of the transaction amounts distribution.
3. **Data Preprocessing**:
   - Normalization of `Amount` and `Time` columns.
   - Splitting of the data into training and testing sets.
   - Use of SMOTE (Synthetic Minority Over-sampling Technique) to balance the imbalanced classes.
4. **Model Building**: 
   - Logistic Regression
   - Decision Tree
   - Random Forest
5. **Model Evaluation**: 
   - Classification report: Precision, Recall, F1-Score.
   - ROC-AUC score and ROC curve visualization.

## Libraries Used

- **pandas**: For data manipulation and analysis.
- **matplotlib** and **seaborn**: For visualizations.
- **scikit-learn**: For building and evaluating machine learning models.
- **imblearn**: For handling imbalanced data with SMOTE.

## Installation

To run this project, you need to have the following Python libraries installed:

```bash
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## How to Run

1. Ensure the dataset `creditcard.csv` is in the same directory as the script.
2. Run the script in a Python environment:

```bash
python fraud_detection.py
```

3. The script will:
   - Perform exploratory data analysis.
   - Preprocess the data by normalizing and handling class imbalance.
   - Train three machine learning models.
   - Evaluate and visualize the performance of each model.

## Dataset

The dataset used for this project contains the following features:
- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **Amount**: The amount of the transaction.
- **Class**: The target variable, where `1` indicates a fraudulent transaction and `0` indicates a normal transaction.

## Data Preprocessing

- The `Amount` and `Time` features are scaled using `StandardScaler`.
- The dataset is split into training (70%) and testing (30%) sets.
- SMOTE is applied to balance the class distribution in the training data.

## Models and Evaluation

Three models are trained and compared:
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**

Each model is evaluated using:
- **Classification report**: Displays Precision, Recall, and F1-Score.
- **ROC-AUC score**: Measures the model's ability to distinguish between the classes.
- **ROC Curve**: A graphical representation of model performance.

### Example Output

```bash
Logistic Regression:
              precision    recall  f1-score   support

           0       0.99      1.00      1.00     85296
           1       0.91      0.63      0.74       148

    accuracy                           0.99     85444
   macro avg       0.95      0.82      0.87     85444
weighted avg       0.99      0.99      0.99     85444

ROC AUC for Logistic Regression: 0.97
```

## Visualization

The following visualizations are generated:
1. **Class distribution** before and after applying SMOTE.
2. **Distribution of transaction amounts**.
3. **ROC curves** for all models, comparing their performance.

## Conclusion

This project demonstrates the use of machine learning techniques for detecting fraudulent transactions. Random Forest achieved the highest ROC-AUC score, making it the best-performing model in this analysis. The use of SMOTE successfully addressed the class imbalance issue, improving the model's ability to detect fraudulent transactions.
