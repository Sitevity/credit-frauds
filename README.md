# Predictive Credit Card Fraud Detection

## 1. Overview
---
This design doc outlines the development of a web application for credit card fraud detection using a real-world dataset. The application will utilize machine learning models that:

- Evaluate whether a transaction is fraudulent or not based on various transaction features.
- Identify the type of fraud in the event of a fraudulent transaction.

## 2. Motivation
---
Credit card fraud is a growing problem that costs businesses and consumers billions of dollars each year. Developing a web application for credit card fraud detection can provide real-time insights into transaction patterns, enabling proactive fraud prevention and reducing financial losses.

## 3. Success Metrics
---
The success of the project will be measured based on the following metrics:

- Precision, recall, and ROC_AUC score of the machine learning models.
- Responsiveness and ease of use of the web application.
- Reduction in financial losses due to fraudulent transactions.

## 4. Requirements & Constraints
---
### 4.1 Functional Requirements

The web application should provide the following functionality:

- Users can provide transaction details to the model and receive a prediction of whether the transaction is fraudulent or not, and the type of fraud.
- Users can view and infer the performance metrics of different machine learning models.
- Users can visualize the data and gain insights into the behavior of fraudulent transactions.

### 4.2 Non-functional Requirements

The web application should meet the following non-functional requirements:

- The model should have high precision, recall, and ROC_AUC Score.
- The web application should be responsive and easy to use.
- The web application should be secure and protect user data.

### 4.3 Constraints

- The application should be built using Streamlit.
- The cost of deployment should be minimal or zero.

### 4.4 Out-of-scope

- Integrating with external applications or data sources.
- Providing detailed transaction diagnostic information.

## 5. Methodology
---
### 5.1. Problem Statement

The problem is to develop a machine learning model that accurately identifies fraudulent credit card transactions.

### 5.2. Data

The dataset consists of over 284,000 credit card transactions, with 28 anonymized features. The target variable is a binary label indicating whether the transaction is fraudulent or not.

### 5.3. Techniques
We will utilize both a binary classification model to predict whether a transaction is fraudulent or not . The following machine learning techniques will be used:

- Data preprocessing and balancing
- Feature engineering and selection
- Model selection and training
- Model evaluation and testing

## 6. Architecture
---
The web application architecture will consist of the following components:

- A frontend web application and deployment-built using Streamlit
- A machine learning model for credit card fraud detection
- Docker containers to run the frontend, backend, and model
- Cloud infrastructure to host the application


## 7. Pipeline

### 7.1. Rough Code
---

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.SVM import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

# Load the data
df = pd.read_csv('../input/creditcard.csv')

# Exploratory Data Analysis
print(f'No Frauds: {round(df["Class"].value_counts()[0]/len(df) * 100,2)}% of the dataset')
print(f'Frauds: {round(df["Class"].value_counts()[1]/len(df) * 100,2)}% of the dataset')

# Handle class imbalance using SMOTE
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train and evaluate models
classifiers = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
    "SVC": SVC(),
    "CatBoostClassifier": CatBoostClassifier(verbose=0),
    "LGBMClassifier": LGBMClassifier()
}

for name, clf in classifiers.items():
    clf.fit(X_train_resampled, y_train_resampled)
    y_pred = clf.predict(X_test)
    
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}")
    print(confusion_matrix(y_test, y_pred))
    print()
```

This project pipeline includes the following steps:

1. **Data Loading**: The credit card transaction dataset is loaded from a CSV file.
2. **Exploratory Data Analysis**: The class imbalance in the dataset is analyzed, with the majority of transactions being non-fraudulent.
4. **Model Training and Evaluation**: Five different classifiers (Logistic Regression, Random Forest, AdaBoost, CatBoost, and LightGBM) are trained on the resampled data, and their performance is evaluated using accuracy, precision, recall, and ROC AUC-score.

The pipeline demonstrates the end-to-end workflow for developing and evaluating a credit card fraud detection system, including data preprocessing, model training, and performance evaluation.

## 8. Conclusion

The web application for credit card fraud detection provides a user-friendly interface for predicting fraudulent transactions and analyzing model performance. By leveraging machine learning techniques and addressing class imbalance, the application can help financial institutions reduce financial losses and improve customer experience.

## 9. Reference

This app was taken inspiration from this [project](https://github.com/vishwasg217/Predictive-Maintenance). You might checking the main notebook I made on Kaggle using [kernels](https://www.kaggle.com/code/luficergfree/voting-for-imbalanced-dataset-with-roc-0-98/)