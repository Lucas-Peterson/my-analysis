# Credit Card Fraud Detection with Machine Learning

## Overview

This project demonstrates the process of detecting fraudulent credit card transactions using machine learning.  
The models are trained on the ["Credit Card Fraud Detection"](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.  
The primary goal is to predict the probability of a transaction being fraudulent.

The workflow focuses on two models:
1. **Logistic Regression**
2. **Random Forest**

Models are evaluated and compared using ROC-AUC scores, classification reports, and ROC curve visualizations. The code also allows you to predict the fraud probability of any custom transaction using both models.

---

## Project Structure

1. **Data Loading**:  
   - The dataset is automatically downloaded using [KaggleHub](https://pypi.org/project/kagglehub/).

2. **Exploratory Data Analysis (EDA)**:  
   - Visualization of class distribution (normal vs. fraudulent transactions)
   - Visualization of transaction amount distribution

3. **Data Preprocessing**:  
   - Normalization of the `Amount` and `Time` columns using `StandardScaler`
   - Splitting the dataset into training and testing sets (70%/30%)

4. **Model Building**:  
   - Logistic Regression  
   - Random Forest (with `class_weight='balanced'`)

5. **Model Evaluation**:  
   - Classification reports (precision, recall, F1-score)
   - ROC-AUC scores
   - ROC curve visualization

6. **Predicting a New Transaction**:  
   - Custom function allows you to input a new transaction and receive fraud probabilities from both models.

---

## Libraries Used

- **pandas**: Data manipulation and analysis
- **matplotlib**, **seaborn**: Visualization
- **scikit-learn**: Machine learning models and metrics
- **kagglehub**: Dataset download automation

---

## Installation

Install all dependencies with:

```bash
pip install pandas matplotlib seaborn scikit-learn kagglehub
