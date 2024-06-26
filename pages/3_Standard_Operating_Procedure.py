import streamlit as st
import requests
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(layout="wide")
st.title("Standard Operating Procedure")


def fetch_pdf_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_file_path = "temp_sop.pdf"
        with open(temp_file_path, "wb") as file:
            file.write(response.content)
        return temp_file_path
    else:
        st.error("Failed to fetch the document.")
        return None


url = "https://github.com/ChansePaskins/CellAnalyzer/raw/main/Cell%20Analysis%20Website%20SOP.pdf"

# Fetch the PDF file from GitHub
file_path = fetch_pdf_from_github(url)
if file_path:
    # Display the PDF using streamlit-pdf-viewer
    pdf_viewer(file_path)
