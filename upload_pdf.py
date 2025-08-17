import requests

file_name = "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"

with open(file_name, 'rb') as f:
    files = {'file': (file_name, f, 'application/pdf')}
    response = requests.post('http://localhost:8080/embed', files=files)

print("Status Code:", response.status_code)
print("Response:", response.text)
