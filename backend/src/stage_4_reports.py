import pandas as pd
import sys
import os
import pickle
from pathlib import Path
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.logger import logger


model = Path("artifacts/model/voting.pkl")
voting_classifier = pickle.load(open(model, 'rb'))
reports_path = Path("artifacts/reports/data_report.html")
preprocessing_train = Path("artifacts/data/processed/train.csv")
preprocessing_test = Path("artifacts/data/processed/test.csv")


def data_report(preprocessing_train, preprocessing_test, model):
    """
    Generate a data quality and model performance report.

    This function reads the preprocessed training and test data, predicts
    probabilities and labels using the provided model, and generates a
    comprehensive report on data quality and model performance. The report is
    saved to the specified reports_path.

    Args:
        preprocessing_train (str): The path to the preprocessed training data file.
        preprocessing_test (str): The path to the preprocessed test data file.
        model (object): The trained model to be used for prediction.

    Returns:
        None
    """
    if os.path.exists(reports_path):
        logger.info(f"The file '{reports_path}'already exists.")
    else:
        ref_data = pd.read_csv(preprocessing_train)
        cur_data = pd.read_csv(preprocessing_test)

        # Predict probabilities using soft voting on train
        original_ypred_train_pred = model.predict_proba(ref_data)[:,1]
        original_ypred_train = [1 if x >= 0.69 else 0 for x in original_ypred_train_pred]
        ref_data["prediction"] = original_ypred_train

        # Predict probabilities using soft voting on test
        original_ypred_test_pred = model.predict_proba(cur_data)[:,1]
        original_ypred_test = [1 if x >= 0.69 else 0 for x in original_ypred_test_pred]
        cur_data["prediction"] = original_ypred_test

        column_map = ColumnMapping()
        column_map.target = 'Class'
        column_map.prediction = 'prediction'
        column_map.numerical_features = ['scaled_amount', 'scaled_time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

        classification_performance_report = Report(metrics=[
            DataDriftPreset(), DataQualityPreset(), ClassificationPreset()
        ])
        classification_performance_report.run(reference_data=ref_data, current_data=cur_data, column_mapping=column_map)
        classification_performance_report.save_html(reports_path)

if __name__ == "__main__":
    logger.info("Starting the Report Section")
    data_report(preprocessing_train, preprocessing_test, voting_classifier)
    logger.info('Data report saved')