import docx
import requests


def read_docx_from_github(url):
    response = requests.get(url)
    with open("Cell Analysis Website SOP.docx", "wb") as file:
        file.write(response.content)
    doc = docx.Document("Cell Analysis Website SOP.docx")
    return doc

