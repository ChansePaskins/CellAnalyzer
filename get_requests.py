import docx
import requests


def read_docx_from_github(url):
    response = requests.get(url)
    with open("temp_sop.docx", "wb") as file:
        file.write(response.content)
    doc = docx.Document("temp_sop.docx")
    return doc

