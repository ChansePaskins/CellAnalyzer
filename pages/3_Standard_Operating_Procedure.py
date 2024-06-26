import streamlit as st
import docx
from get_requests import read_docx_from_github

# Page styling and title
st.set_page_config(layout="wide")
st.title("Cell Counter and Area Analyzer Pipeline")

url = "https://github.com/ChansePaskins/CellAnalyzer/raw/main/Cell%20Analysis%20Website%20SOP.docx"
doc = read_docx_from_github(url)

for para in doc.paragraphs:
    st.write(para.text)