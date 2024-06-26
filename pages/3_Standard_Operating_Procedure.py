import streamlit as st
import docx
from get_requests import read_docx_from_github

# Page styling and title
st.set_page_config(layout="wide")
st.title("Cell Counter and Area Analyzer Pipeline")

url = "https://raw.githubusercontent.com/your_username/your_repo/main/sop.docx"
doc = read_docx_from_github(url)

for para in doc.paragraphs:
    st.write(para.text)