import requests

url = "https://github.com/jtranberg/8_class_model/releases/download/v1.0/best_model.keras"
response = requests.get(url, allow_redirects=True)

print("Status:", response.status_code)
print("Content-Type:", response.headers.get("Content-Type"))
print("Length:", len(response.content))
