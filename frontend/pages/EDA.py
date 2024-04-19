import streamlit as st
import pandas as pd
import sys
import os
# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@st.cache_data()
def run_eda():
    st.title("EDA Questions")
    st.write("1. What is the distribution of the 'Class' (target) in the dataset? How many instances have Fraud and how many have not?")
    st.write("2. But what if we use the original dataset for training?Terrible. Let's see correlation for that matter")
    st.write('3. What is the most & least important features in the dataset to predict our class?')
    st.write('4. Are the least important features also important to keep them in the dataset first place?Or they should be removed in the dataset?')
    st.write("5. Can we see the correlation between the negatively impacting features and the target column?")
    st.write("6. Can we see the correlation between the positively impacting features and the target column?")
    st.divider()

    st.header('Question 1')
    st.write("What is the distribution of the 'Class' label in the dataset? How many instances have Fraud and how many have not?")
    col1, col2 = st.columns(2)
    with col1:
        st.image('artifacts/eda/q1.png', caption="Original dataset")

    with col2:
        st.image('artifacts/eda/q1_1.png', caption = "Resampled dataset")
    st.markdown("""
    > **Note:**  Notice how imbalanced is our original dataset! Most of the transactions are non-fraud. If we use this dataframe as the base for our predictive models and analysis we might get a lot of errors and our algorithms will probably overfit since it will "assume" that most transactions are not fraud. But we don't want our model to assume, we want our model to detect patterns that give signs of fraud!
    - Once we determine how many instances are considered **fraud transactions**.. so we will use balanced dataset for further EDA""")
    st.divider()

    st.header('Question 2')
    st.write("But what if we use the original dataset for training?Terrible. Let's see correlation for that matter")
    st.image('artifacts/eda/q2.png')
    st.image('artifacts/eda/q2_2.webp')
    st.markdown("""
    > **Note:**  Notice how imbalanced is our original dataset! Most of the transactions are non-fraud. If we use this dataframe as the base for our predictive models and analysis we might get a lot of errors and our algorithms will probably overfit since it will "assume" that most transactions are not fraud. But we don't want our model to assume, we want our model to detect patterns that give signs of fraud!So, let's go further for EDA with balanced dataset""")
    st.divider()

    st.header('Question 3')
    st.write('What is the most & least important features in the dataset to predict our class?')
    st.image('artifacts/eda/q3.png')
    st.markdown("""
                - Here, as you can see V14, V10 & V3 are having most of the importance.
                - Here ,also V25, V24, V26 & V28 are having least of the importance.
                """)
    st.divider()

    st.header("Question 4")
    st.write('Are the least important features also important to keep them in the dataset first place?Or they should be removed in the dataset?')
    st.write("So let's check with the Null Hypothesis and Alternate Hypothesis WIth T-test")
    st.latex(r"""
        Null ~Hypothesis~:-~There~is~no~significant~relationship~between~the~features(V24,V25,V26,V28)~and~target~column~'Class'.
        """)
    st.latex(r"""
             Alternate~Hypothesis~:-~There~is~a~significant~relationship~between~the~features(V24,V25,V26,V28)~and~target~column~'Class'.
             """)
    # Display the DataFrame
    st.dataframe(pd.read_csv('artifacts/eda/q4.csv'))
    st.markdown("""
                - Here, as you can see we can reject the Null Hypothesis easily..
                """)
    st.divider()

    st.header("Question 5")
    st.write("Can we see the correlation between the negatively impacting features and the target column?")
    st.image('artifacts/eda/q5.png')
    st.markdown("### Looks like they are having relationship between class , you can easily do that with watching class '1' has lower values than class '0'..so ,it's there")
    st.divider()

    st.header("Question 6")
    st.write("Can we see the correlation between the positively impacting features and the target column?")
    st.image('artifacts/eda/q6.png')
    st.markdown("### Looks like they are having relationship between class , you can easily do that with watching class '1' has higher values than class '0'..so ,it's there")

if __name__ == "__main__":
    run_eda()