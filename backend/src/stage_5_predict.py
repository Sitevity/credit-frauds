import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.logger import logger


img_path = Path('artifacts/img.png')
model = Path("artifacts/model/voting.pkl")
preprocessing_test = Path('artifacts/data/processed/test.csv')
original = Path("artifacts/data/raw/creditcard.csv")
bal_df = Path('artifacts/balanced_df.csv')
voting_classifier = pickle.load(open(model, 'rb'))
seed = 42


def confusion(preprocessing_test, model):
    """
    Generate a confusion matrix plot for the model's performance on the test data.

    This function reads the preprocessed test data, predicts the labels using the
    provided model, and generates a confusion matrix plot. The plot is saved to
    the specified img_path.

    Args:
        preprocessing_test (str): The path to the preprocessed test data file.
        model (object): The trained model to be used for prediction.

    Returns:
        None
    """
    if os.path.exists(img_path):
        logger.info(f"The file '{img_path}' already exists.")
    else:
        preprocessing_test = pd.read_csv(preprocessing_test)
        original_ytest = preprocessing_test['Class']
        original_Xtest = preprocessing_test.drop('Class', axis=1)
        original_ypred_test_pred = model.predict_proba(original_Xtest)[:, 1]
        original_ypred_test = [1 if x >= 0.69 else 0 for x in original_ypred_test_pred]

        cm = confusion_matrix(original_ytest, original_ypred_test)
        fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 5))
        sns.heatmap(cm, annot=True, ax=ax1, fmt='d', cmap="Blues")
        ax1.set_title('Testing Confusion Matrix', fontsize=14)
        plt.savefig(img_path)

def balanced_df(data):
    """
    Create a balanced dataset by sampling an equal number of fraud and non-fraud instances from the given data.

    Args:
        data (str): The path to the input CSV file containing the credit card transaction data.

    Returns:
        None
        
    Side Effects:
        - Reads the input CSV file and creates a balanced dataset.
        - Saves the balanced dataset to a new CSV file named 'bal_df' in the current working directory.
    """
    if os.path.exists(bal_df):
        logger.info(f"The file '{bal_df}' already exists.")
    
    else:
        df = pd.read_csv(data)
        df = df.sample(frac=1, random_state=seed)
        fraud_df = df.loc[df['Class'] == 1].sample(50, random_state=seed)
        non_fraud_df = df.loc[df['Class'] == 0].sample(50, random_state=seed)
        normal_distributed_df = pd.concat([fraud_df, non_fraud_df], axis=0)
        new_df = normal_distributed_df.sample(frac=1, random_state=seed)
        new_df.to_csv(bal_df, index=False)


if __name__ == "__main__":
    logger.info('Confusion Prediction Started...')
    confusion(preprocessing_test, voting_classifier)
    balanced_df(original)
    logger.info('Confusion Prediction Ended..')