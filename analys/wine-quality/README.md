# Wine Classification using RandomForest

This project demonstrates the use of the `RandomForestClassifier` from the `scikit-learn` library to classify wines based on their chemical properties. The dataset used is the built-in `wine` dataset from `sklearn.datasets`, which contains 178 samples of wine categorized into three different classes. The goal of the project is to build a classification model, evaluate its performance using various metrics, and visualize the results.

## Features
- **Random Forest Classification**: A powerful ensemble method for classification.
- **Performance Metrics**: Accuracy, Confusion Matrix, Precision, Recall, F1-score, and ROC-AUC curves.
- **Cross-validation**: Used to evaluate the model's performance on different splits of the dataset.
- **Feature Importance**: Identify which features (chemical properties of the wine) are most important for the model's predictions.
- **Visualizations**: Matrices and curves for better understanding of the model's performance.




## Project Structure

# wine_classification.py: 
    The main Python script that contains the code for loading the dataset, training the model, and visualizing the results.

# README.md: 
    This file, providing an overview of the project.

# Dataset
    The dataset is loaded using sklearn.datasets.load_wine(). It contains 178 samples of wine, each described by 13 different chemical features such as alcohol, malic acid, and   
    magnesium, and classified into one of three wine categories.

# Dataset Structure:
    Features: 13 numerical features representing the chemical properties of the wines.
    Target Classes: 3 classes representing different types of wines.




## Installation
1. Clone the repository (or download the project files)
git clone https://github.com/your-username/wine-classification.git

 2. Navigate into the project directory
cd wine-classification

3. (Optional but recommended) Create and activate a virtual environment
 On Windows:
python -m venv venv
venv\Scripts\activate

On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

4. Install the required dependencies from the requirements.txt file
pip install -r requirements.txt

5. Run the project
python wine_classification.py


## Output
After running the script, the following outputs will be generated:

Accuracy Score: A numerical value representing how well the model performs on the test data.
Confusion Matrix: A matrix visualizing how many samples were correctly and incorrectly classified.
Classification Report: Precision, Recall, and F1-score for each class of wine.
Feature Importance Plot: A bar plot showing the importance of each feature for making predictions.
ROC Curves: ROC curves and AUC values for each wine class, visualizing the trade-off between True Positive Rate and False Positive Rate.





## Example output

Accuracy: 0.98

Confusion Matrix:
[[19  0  0]
 [ 0 18  0]
 [ 0  1 16]]

Classification Report:
              precision    recall  f1-score   support

    class_0       1.00      1.00      1.00        19
    class_1       0.95      1.00      0.97        18
    class_2       1.00      0.94      0.97        17

   accuracy                           0.98        54
  macro avg       0.98      0.98      0.98        54
weighted avg       0.98      0.98      0.98        54





## Visualizations
The script generates the following visualizations:

### Confusion Matrix:
Heatmap of predicted vs actual classifications.

### Feature Importance Plot:
Shows which features are most important for predicting wine class.

### ROC Curves:
ROC curves for each class, showing the model's performance in terms of True Positive Rate and False Positive Rate.




## Improvements
Possible future improvements to this project:

Hyperparameter Tuning: Experiment with different hyperparameters of the RandomForestClassifier to optimize performance.

Additional Models: Try other classification models (e.g., SVM, KNeighborsClassifier) for comparison.

More Metrics: Implement more advanced metrics like Matthews correlation coefficient or Cohen's kappa to better understand performance.
