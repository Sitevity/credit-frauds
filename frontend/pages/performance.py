import streamlit as st
import pandas as pd
import sys
import os
# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_performance():
    st.title("Performance Measure")
    st.header("MODEL Comparison")
    st.write("Models Table")
    st.dataframe(pd.read_csv('artifacts/scores.csv'))
    st.divider()
    st.latex("Best~Model~:~Catboost~-~0.858~Score")
    st.divider()
    st.header("How model performs on random(imbalanced) dataset??")
    st.write("Confusion Matrix")
    st.image('artifacts/img.png')
    st.write('So, it does good job in classification, let go towards report..')
    st.write("Classification Report")
    st.dataframe(pd.read_csv('artifacts/report.csv'))

if __name__ == "__main__":
    run_performance()