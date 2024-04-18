import streamlit as st
import pandas as pd
import pickle
import random
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import sys
import os
# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

model = Path("artifacts/model/voting.pkl")
balanced_df = Path('artifacts/balanced_df.csv')

def run_predict():
    voting_classifier = pickle.load(open(model, 'rb'))
    st.title("Credit Card Fraud Detection")
    st.write("""
    This app will help you to detect credit card fraud using a machine learning model(Catboost).
    The model was trained using [Kaggle Credit Craud Dataset](#https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) .
    """)
    df = pd.read_csv(balanced_df)
    st.dataframe(df)

    # Define the choice for the user
    choice = st.radio("Choose an option:", ("Select a Random or Your example", "Enter your own inputs"))

    if choice == "Select a Random or Your example":

        example = st.button("Take a Random example", type="primary")

        if example:
            # generate a random number between 0 and 74
            num = random.randint(0, 74)
            st.write("This is the testing example:",num)
            df_example = df.iloc[num]

        else:
            # user specified a random number
            num = st.number_input("Enter a random number between 0 and 99", step = 1, min_value = 0, max_value = 99)
            df_example = df.iloc[num]

        amount = st.number_input('Amount', step = 1.0, value=df_example.Amount)
        time = st.number_input('Time', step=1.0, value = df_example.Time)
        V1 = st.number_input('V1', value = df_example.V1)
        V2 = st.number_input('V2', value = df_example.V2)
        V3 = st.number_input('V3', value = df_example.V3)
        V4 = st.number_input('V4', value = df_example.V4)
        V5 = st.number_input('V5', value = df_example.V5)
        V6 = st.number_input('V6', value = df_example.V6)
        V7 = st.number_input('V7', value = df_example.V7)
        V8 = st.number_input('V8', value = df_example.V8)
        V9 = st.number_input('V9', value = df_example.V9)
        V10 = st.number_input('V10', value = df_example.V10)
        V11 = st.number_input('V11', value = df_example.V11)
        V12 = st.number_input('V12', value = df_example.V12)
        V13 = st.number_input('V13', value = df_example.V13)
        V14 = st.number_input('V14', value = df_example.V14)
        V15 = st.number_input('V15', value = df_example.V15)
        V16 = st.number_input('V16', value = df_example.V16)
        V17 = st.number_input('V17', value = df_example.V17)
        V18 = st.number_input('V18', value = df_example.V18)
        V19 = st.number_input('V19', value = df_example.V19)
        V20 = st.number_input('V20', value = df_example.V20)
        V21 = st.number_input('V21', value = df_example.V21)
        V22 = st.number_input('V22', value = df_example.V22)
        V23 = st.number_input('V23', value = df_example.V23)
        V24 = st.number_input('V24', value = df_example.V24)
        V25 = st.number_input('V25', value = df_example.V25)
        V26 = st.number_input('V26', value = df_example.V26)
        V27 = st.number_input('V27', value = df_example.V27)
        V28 = st.number_input('V28', value = df_example.V28)

    else:
        amount = st.number_input('Amount', step=1.0)
        time = st.number_input('Time', step=1.0)
        V1 = st.number_input('V1')
        V2 = st.number_input('V2')
        V3 = st.number_input('V3')
        V4 = st.number_input('V4')
        V5 = st.number_input('V5')
        V6 = st.number_input('V6')
        V7 = st.number_input('V7')
        V8 = st.number_input('V8')
        V9 = st.number_input('V9')
        V10 = st.number_input('V10')
        V11 = st.number_input('V11')
        V12 = st.number_input('V12')
        V13 = st.number_input('V13')
        V14 = st.number_input('V14')
        V15 = st.number_input('V15')
        V16 = st.number_input('V16')
        V17 = st.number_input('V17')
        V18 = st.number_input('V18')
        V19 = st.number_input('V19')
        V20 = st.number_input('V20')
        V21 = st.number_input('V21')
        V22 = st.number_input('V22')
        V23 = st.number_input('V23')
        V24 = st.number_input('V24')
        V25 = st.number_input('V25')
        V26 = st.number_input('V26')
        V27 = st.number_input('V27')
        V28 = st.number_input('V28')

    predict = st.button("Predict", type="primary")

    if predict:
        # Create a DataFrame from the input data
        data = {
            'Amount': [amount], 'Time': [time],
            'V1': [V1], 'V2': [V2], 'V3': [V3], 'V4': [V4], 'V5': [V5], 'V6': [V6],
            'V7': [V7], 'V8': [V8], 'V9': [V9], 'V10': [V10], 'V11': [V11], 'V12': [V12],
            'V13': [V13], 'V14': [V14], 'V15': [V15], 'V16': [V16], 'V17': [V17],
            'V18': [V18], 'V19': [V19], 'V20': [V20], 'V21': [V21], 'V22': [V22],
            'V23': [V23], 'V24': [V24], 'V25': [V25], 'V26': [V26], 'V27': [V27],
            'V28': [V28]
        }
        df_example = pd.DataFrame(data)

        rob_scaler = RobustScaler()
        rob_scaler.fit(df['Amount'].values.reshape(-1, 1))
        rob_scaler.fit(df['Time'].values.reshape(-1, 1))
        df_example['scaled_amount'] = rob_scaler.transform(df_example['Amount'].values.reshape(-1, 1))
        df_example['scaled_time'] = rob_scaler.transform(df_example['Time'].values.reshape(-1, 1))
        df_example.drop(['Time', 'Amount'], axis=1, inplace=True)

        # Predict the output using the loaded model
        prediction_proba = voting_classifier.predict_proba(df_example)[:, 1]
        prediction = [1 if prediction_proba[0] >= 0.69 else 0]

        if prediction[0] == 0:
            st.write('Prediction: This transaction is not fraudulent.')
        else:
            st.write('Prediction: This transaction is fraudulent.')
            st.write('Fraud Probability:', prediction_proba[0])

if __name__ == "__main__":
    run_predict()