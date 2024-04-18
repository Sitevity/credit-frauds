import pandas as pd
import pickle
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.logger import logger
from backend.src.stage_1_data import downloading_data, preprocessing_data
from backend.src.stage_2_eda import q1, q2, q3, q4, q5, q6
from backend.src.stage_3_train import model
from backend.src.stage_4_reports import data_report
from backend.src.stage_5_predict import confusion, balanced_df

preprocessing_train = Path("artifacts/data/processed/train.csv")
preprocessing_test = Path("artifacts/data/processed/test.csv")
data_file = Path("artifacts/data/raw/creditcard.csv")
model_path = Path('artifacts/model/voting.pkl')
img_dir = Path("artifacts/eda")
voting_classifier = pickle.load(open(model_path, 'rb'))
seed = 42 # Answer of everything(Meaning of life)

def whole_backend():

    logger.info("Starting the data downloading and preprocessing stage...")
    downloading_data(data_file)
    preprocessing_data(preprocessing_train,preprocessing_test, data_file)
    logger.info("Data downloading and preprocessing completed.")

    logger.info("Starting the EDA stage...")
    df = pd.read_csv(data_file)
    df = df.sample(frac=1, random_state = seed)
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0].sample(500, random_state = seed)
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df], axis = 0)
    new_df = normal_distributed_df.sample(frac=1, random_state=seed)

    q1(df, new_df, img_dir)
    q2(df, new_df, img_dir)
    q3(new_df, img_dir)
    q4(new_df, img_dir)
    q5(new_df, img_dir)
    q6(new_df, img_dir)
    logger.info("EDA stage completed.")

    logger.info("Starting the model training stage...")
    train_df = pd.read_csv(preprocessing_train)
    test_df = pd.read_csv(preprocessing_test)
    model(train_df, test_df)
    logger.info("Model training stage completed.")

    logger.info("Starting the Report Section")
    data_report(preprocessing_train, preprocessing_test, voting_classifier)
    logger.info('Data report saved')

    logger.info('Confusion Prediction Started...')
    confusion(preprocessing_test, voting_classifier)
    balanced_df(df)
    logger.info('Confusion Prediction Ended..')

    logger.info("All stages completed.")

if __name__ == "__main__":
    whole_backend()