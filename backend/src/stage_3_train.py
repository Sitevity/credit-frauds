import mlflow
import mlflow.sklearn
import pandas as pd
import os
import sys
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.logger import logger


classifiers = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "SVC": SVC(probability = True),
    "CatBoost": CatBoostClassifier(verbose=0),
    "LGBM": LGBMClassifier()
}
train_df = Path("artifacts/data/processed/train.csv")
test_df = Path("artifacts/data/processed/test.csv")
score = Path('artifacts/scores.csv')
report = Path('artifacts/report.csv')
model_path = Path('artifacts/model/voting.pkl')


def model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train and evaluate multiple classification models on the credit card fraud dataset.

    This function reads the preprocessed training and test data, imputes missing values using
    SimpleImputer, trains and evaluates multiple classification models (Logistic Regression,
    Random Forest, SVC, CatBoost, and LGBM), logs the model performance metrics using MLflow,
    and saves the best performing model, its evaluation metrics, and the classification report.

    Args:
        train_df (pd.DataFrame): The preprocessed training data.
        test_df (pd.DataFrame): The preprocessed test data.

    Returns:
        None
    """
    if (os.path.exists(score) and os.path.exists(report) and os.path.exists(model_path)):
        logger.info(f"The file '{score}' ,'{model_path}' and '{report}' already exists.")
    else:
        X_train = train_df.drop(["Class"], axis=1)
        y_train = train_df["Class"]
        X_test = test_df.drop(["Class"], axis=1)
        y_test = test_df["Class"]

        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        logger.info('loaded..')

        scores = []
        for key, classifier in classifiers.items():
            classifier.fit(X_train, y_train)
            original_y_pred_test = classifier.predict_proba(X_test)[:, 1]
            test_roc_auc = roc_auc_score(y_test, original_y_pred_test)
            original_y_pred_test = [1 if x >= 0.5 else 0 for x in original_y_pred_test]
            test_accuracy = accuracy_score(y_test, original_y_pred_test)
            test_recall = recall_score(y_test, original_y_pred_test)
            test_precision = precision_score(y_test, original_y_pred_test)
            test_f1 = f1_score(y_test, original_y_pred_test)

            with mlflow.start_run():
                mlflow.sklearn.log_model(classifier, key)
                mlflow.log_metric("Accuracy", test_accuracy)
                mlflow.log_metric("Precision", test_precision)
                mlflow.log_metric("Recall", test_recall)
                mlflow.log_metric("ROC-AUC", test_roc_auc)
                mlflow.log_metric("F1", test_f1)

            scores.append([test_accuracy, test_precision, test_recall, test_roc_auc, test_f1])

        scores_df = pd.DataFrame(
            columns=["Model", "Accuracy", "Precision", "Recall", "ROC-AUC", "F1"],
            data=[(k, *v) for k, v in zip(classifiers.keys(), scores)]
        )

        best_model_idx = scores_df["F1"].idxmax()
        best_model = list(classifiers.values())[best_model_idx]

        y_pred = best_model.predict(X_test)
        reports = classification_report(y_test, y_pred, output_dict=True)
        reports = pd.DataFrame(reports).transpose()

        scores_df.to_csv(score, index=False)
        reports.to_csv(report, index=False)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

if __name__ == "__main__":
    train_df = pd.read_csv(train_df)
    test_df = pd.read_csv(test_df)
    model(train_df, test_df)
    