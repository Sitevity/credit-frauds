import streamlit as st
import sys
import os
# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_reports():
    with ('artifacts/reports/data_report.html', 'rb') as f:
        html_content = f.read()

    st.components.v1.iframe(html_content, width=800, height=600)

if __name__ == "__main__":
    run_reports()