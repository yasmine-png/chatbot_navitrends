import requests

url = "http://localhost:8000/chatbot/"
payload = {"question": "M’aiderez-vous à indexer mon site sur Google ?"}
response = requests.post(url, json=payload)
print("Status code:", response.status_code)
print("Response JSON:", response.json())
