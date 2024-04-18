import warnings
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.logger import logger
warnings.filterwarnings("ignore")


seed = 42
data_file = Path("artifacts/data/raw/creditcard.csv")
colors = ["#0101DF", "#DF0101"]
img_dir = Path("artifacts/eda")


def q1(df, new_df, img_dir):
    """
    Perform analysis on the class imbalance in the given DataFrame.

    Args:
        df (pandas.DataFrame): The original DataFrame.
        new_df (pandas.DataFrame): The balanced DataFrame, with 'Class' column.
        img_dir (str): The directory to save the image.

    Returns:
        None
    """
    new_img_path = os.path.join(img_dir, 'q1_1.png')
    img_path = os.path.join(img_dir, 'q1.png')
    
    if os.path.exists(new_img_path) and os.path.exists(img_path):
        logger.info(f"The file '{img_path}' and '{new_img_path}' already exists.")

    else:
        # Plot the histogram for df
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Class', data=df, palette=colors)
        plt.title('Class Distributions (Original DataFrame)', fontsize=14)
        plt.savefig(img_path)
        plt.close()

        # Plot the histogram for new_df
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Class', data=new_df, palette=colors)
        plt.title('Class Distributions (Balanced DataFrame)', fontsize=14)
        plt.savefig(new_img_path)
        plt.close()
    
def q2(df, new_df, img_dir):
    """
    Analyze the correlation matrix of the entire dataset and a subsampled dataset.
    
    Args:
        df (pandas.DataFrame): The original, imbalanced DataFrame.
        new_df (pandas.DataFrame): The subsampled, balanced DataFrame.
    
    Returns:
        None
    """
    img_path = os.path.join(img_dir, 'q2.png')
    if os.path.exists(img_path):
        logger.info(f"The file '{img_path}' already exists.")

    else:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))

        # Plot the correlation matrix for the entire DataFrame
        corr = df.corr()
        sns.heatmap(corr, cmap='coolwarm_r', ax=ax1)
        ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

        # Plot the correlation matrix for the subsampled DataFrame
        sub_sample_corr = new_df.corr()
        sns.heatmap(sub_sample_corr, cmap='coolwarm_r', ax=ax2)
        ax2.set_title('Subsampled Correlation Matrix \n (use for reference)', fontsize=14)
        plt.savefig(img_path)

def q3(new_df, img_dir):
    """
    Analyze which columns(features) is the most and least correlated with the target using Random Forest feature importance
    
    Args:
        new_df (pandas.DataFrame): The balanced DataFrame.
        img_dir (str): The directory to save the image.
    
    Returns:
        None
    """
    img_path = os.path.join(img_dir, 'q3.png')
    if os.path.exists(img_path):
        logger.info(f"The file '{img_path}' already exists.")

    else:
        clf = RandomForestClassifier()
        target = 'Class'
        predictors = ['scaled_time', 'scaled_amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

        final_tmp = pd.DataFrame()
        # Execute the code 10 times
        for i in range(10):
            clf.fit(new_df[predictors], new_df[target].values)
            # Execute your code here
            tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
            tmp = tmp.sort_values(by='Feature importance', ascending=False)
            
            # Concatenate the current results to the final DataFrame
            final_tmp = pd.concat([final_tmp, tmp], ignore_index=True)

        # Plot the final result
        plt.figure(figsize=(10, 7))
        plt.title('Features importance', fontsize=14)
        s = sns.barplot(x='Feature', y='Feature importance', data=final_tmp)
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
        plt.savefig(img_path)

def q4(new_df, img_dir):
    """
    Perform a T-test on least important columns (V25, V24, V26 & V28) with Class column and save result in dataframe

    Args:
        new_df (pandas.DataFrame): The balanced DataFrame.
        csv_file (str): The path to the CSV file to save the results.

    Returns:
        None
    """
    csv_file = os.path.join(img_dir, 'q4.csv')
    if os.path.exists(csv_file):
        logger.info(f"The file '{csv_file}' already exists.")

    else:
        # Perform t-test for each column
        t_values = []
        p_values = []
        results = []

        for col in ['V25', 'V24', 'V26', 'V28']:
            t_stat, p_val = ttest_ind(new_df[col], new_df['Class'])
            t_values.append(t_stat)
            p_values.append(p_val)
            
            # Determine if we reject or accept the null hypothesis
            if p_val < 0.05:
                results.append('Reject Null the Hypothesis')
            else:
                results.append('Accept Null the Hypothesis')

        result_table = pd.DataFrame({
            'Column Name': ['V25', 'V24', 'V26', 'V28'],
            't-value': t_values,
            'p-value': p_values,
            'Reject Null or Accept Null': results
        })
        result_table.to_csv(csv_file, index = False)

def q5(new_df, img_dir):
    """
    Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction

    Args:
        new_df (pandas.DataFrame): The balanced DataFrame.
        img_dir (str): The directory to save the image.

    Returns:
        None
    """

    img_path = os.path.join(img_dir, 'q5.png')
    if os.path.exists(img_path):
        logger.info(f"The file '{img_path}' already exists.")

    else:
        f, axes = plt.subplots(ncols=4, figsize=(20,4))
        boxprops = dict(linestyle='-', linewidth=2, color='black')
        # Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
        sns.boxplot(x="Class", y="V17", data=new_df, palette=colors, ax=axes[0], boxprops=boxprops)
        axes[0].set_title('V17 vs Class Negative Correlation')

        sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1], boxprops=boxprops)
        axes[1].set_title('V14 vs Class Negative Correlation')


        sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2], boxprops=boxprops)
        axes[2].set_title('V12 vs Class Negative Correlation')


        sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3], boxprops=boxprops)
        axes[3].set_title('V10 vs Class Negative Correlation')
        plt.savefig(img_path)

def q6(new_df, img_dir):
    """
    Posistive Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction

    Args:
        new_df (pandas.DataFrame): The balanced DataFrame.
        img_dir (str): The directory to save the image.

    Returns:
        None
    """

    img_path = os.path.join(img_dir, 'q6.png')
    if os.path.exists(img_path):
        logger.info(f"The file '{img_path}' already exists.")

    else:
        f, axes = plt.subplots(ncols=4, figsize=(20,4))
        # Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
        sns.boxplot(x="Class", y="V11", data=new_df, palette=colors, ax=axes[0])
        axes[0].set_title('V11 vs Class Positive Correlation')

        sns.boxplot(x="Class", y="V4", data=new_df, palette=colors, ax=axes[1])
        axes[1].set_title('V4 vs Class Positive Correlation')

        sns.boxplot(x="Class", y="V2", data=new_df, palette=colors, ax=axes[2])
        axes[2].set_title('V2 vs Class Positive Correlation')

        sns.boxplot(x="Class", y="V19", data=new_df, palette=colors, ax=axes[3])
        axes[3].set_title('V19 vs Class Positive Correlation')
        plt.savefig(img_path)


if __name__ == "__main__":
    logger.info('EDA started...')
    df = pd.read_csv(data_file)
    df = df.sample(frac=1, random_state = seed)

    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0].sample(500, random_state = seed)

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df], axis = 0)
    # Shuffle dataframe rows
    new_df = normal_distributed_df.sample(frac=1, random_state=seed)

    q1(df, new_df, img_dir)
    q2(df, new_df, img_dir)
    q3(new_df, img_dir)
    q4(new_df, img_dir)
    q5(new_df, img_dir)
    q6(new_df, img_dir)

    logger.info('EDA completed successfully!')