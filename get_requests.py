import requests


def read_pdf_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_file_path = "temp_sop.pdf"
        with open(temp_file_path, "wb") as file:
            file.write(response.content)
        return temp_file_path


