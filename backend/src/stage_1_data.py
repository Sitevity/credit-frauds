import os
import subprocess
import zipfile
import sys
from sklearn.preprocessing import RobustScaler
import pandas as pd
from pathlib import Path
# Add the parent directory of 'config' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.logger import logger
from sklearn.model_selection import StratifiedKFold


seed = 42
preprocessing_train = Path("artifacts/data/processed/train.csv")
preprocessing_test = Path("artifacts/data/processed/test.csv")
data_file = Path("artifacts/data/raw/creditcard.csv")


def downloading_data(data_file):
    """
    Download and preprocess a Kaggle dataset.

    This function checks if the specified data file already exists. If not, it
    sets the Kaggle API credentials, downloads the dataset, unzips the
    downloaded file, and cleans up the downloaded zip file.

    Args:
        data_file (str): The path to the data file.

    Raises:
        SystemExit: If the Kaggle API credentials are not updated with real
            values, or if the 'kaggle.json' file does not exist in the '.kaggle'
            directory.
    """
    # Check if the file already exists
    if os.path.exists(data_file):
        logger.info(f"The file '{data_file}' already exists.")

    else:
        # Step 1: Set Kaggle API credentials
        # Check if the kaggle.json file exists in the .kaggle directory
        base_dir = os.path.dirname(__file__)
        kaggle_dir = os.path.join(base_dir, ".kaggle")
        kaggle_json_file = os.path.join(kaggle_dir, "kaggle.json")
        if os.path.exists(kaggle_json_file):
            # Read the kaggle.json file
            with open(kaggle_json_file, "r") as f:
                kaggle_json_content = f.read()
            
            # Check if the kaggle.json content is not the same as the default
            default_kaggle_json = '''
            {
                "username": "your_username",
                "key": "your_api_key"
            }
            '''
            if kaggle_json_content.strip() != default_kaggle_json.strip():
                # Save the Kaggle API credentials to the kaggle.json file
                if not os.path.exists(kaggle_dir):
                    os.makedirs(kaggle_dir)
                with open(kaggle_json_file, "w") as f:
                    f.write(kaggle_json_content)
                
                # Set permissions for the credentials file
                os.chmod(kaggle_json_file, 0o600)
            else:
                logger.error("You haven't updated the Kaggle API credentials with real values in 'kaggle.json'.")
                sys.exit(1)
        else:
            logger.error("The 'kaggle.json' file does not exist in the '.kaggle' directory.")
            sys.exit(1)

        # Step 3: Download dataset
        dataset_name = "mlg-ulb/creditcardfraud"
        download_command = f"kaggle datasets download -d {dataset_name} -p artifacts/data/raw"
        subprocess.run(download_command.split())

        # Step 4: Unzip the dataset
        os.chdir("artifacts/data/raw")
        zip_file = f"{dataset_name.split('/')[-1]}.zip"
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")

        # Step 5: Cleanup: remove downloaded zip file
        os.remove(zip_file)

def preprocessing_data(preprocessing_train , preprocessing_test, data_file):
    """
    Preprocess the credit card fraud dataset.

    This function reads the raw dataset from the specified data_file, applies
    RobustScaler to the 'Time' and 'Amount' columns, drops the 'Time' and
    'Amount' columns, splits the data into training and test sets using
    StratifiedKFold, and saves the preprocessed data to the specified
    preprocessing_train and preprocessing_test files.

    Args:
        preprocessing_train (str): The path to the preprocessed training data file.
        preprocessing_test (str): The path to the preprocessed test data file.
        data_file (str): The path to the raw data file.

    Returns:
        None
    """
    # Check if the file already exists
    if os.path.exists(preprocessing_train) and os.path.exists(preprocessing_test):
        logger.info(f"The file '{preprocessing_train}' and '{preprocessing_test}' already exists.")

    else:
        df = pd.read_csv(data_file)
        # RobustScaler is less prone to outliers.
        rob_scaler = RobustScaler()

        df.insert(0, 'scaled_amount', rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1)))
        df.insert(1, 'scaled_time', rob_scaler.fit_transform(df['Time'].values.reshape(-1,1)))
        df.drop(['Time','Amount'], axis=1, inplace=True)

        X = df.drop('Class', axis=1)
        y = df['Class']
        sss = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

        for train_index, test_index in sss.split(X, y):
            original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
            original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

        # Turn into an array
        # Convert numpy arrays to pandas DataFrame
        original_Xtrain = pd.DataFrame(original_Xtrain.values, columns=X.columns)
        original_Xtest = pd.DataFrame(original_Xtest.values, columns=X.columns)
        original_ytest = pd.DataFrame(original_ytest.values, columns=['Class'])
        original_ytrain = pd.DataFrame(original_ytrain.values, columns=['Class'])

        train = pd.concat([original_Xtrain, original_ytrain], axis=1)
        test = pd.concat([original_Xtest, original_ytest], axis=1)

        train.to_csv(preprocessing_train, index=False)
        test.to_csv(preprocessing_test, index=False)



if __name__ == "__main__":
    logger.info('Data downloading, loading & preprocessing started...')
    downloading_data(data_file)
    preprocessing_data(preprocessing_train, preprocessing_test, data_file)
    logger.info('Data downloading, loading & preprocessing completed successfully!')