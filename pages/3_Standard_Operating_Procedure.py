import streamlit as st
from get_requests import read_pdf_from_github
import fitz

# Page styling and title
st.set_page_config(layout="wide")
st.title("Cell Counter and Area Analyzer Pipeline")


def display_pdf(file_path):
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        st.write(text)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            st.image(image_bytes)


url = "https://github.com/ChansePaskins/CellAnalyzer/raw/main/Cell%20Analysis%20Website%20SOP.pdf"
doc = read_pdf_from_github(url)
display_pdf(doc)
