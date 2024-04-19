import streamlit as st
import os
import chardet

def run_reports():
    html_file = os.path.join(os.getcwd(), 'artifacts', 'reports', 'data_report.html')
    
    # Try to open the file with UTF-8 encoding
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, use chardet to detect the encoding
        with open(html_file, 'rb') as f:
            html_content_bytes = f.read()
            result = chardet.detect(html_content_bytes)
            encoding = result['encoding']
            html_content = html_content_bytes.decode(encoding)
     # Wrap the HTML content in a scrollable container
    scrollable_html = f"""
        <div style="max-height: 500px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;">
            {html_content}
        </div>
    """

    # Render the scrollable HTML content in Streamlit
    st.components.v1.html(scrollable_html, height=800)

if __name__ == "__main__":
    run_reports()