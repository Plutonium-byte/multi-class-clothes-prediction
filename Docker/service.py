import requests

url = "http://localhost:5000/predict"
image_path = "path/to/image"

try:
    with open(image_path, "rb") as img:
        files = {"image": img}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Server Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"Connection Error: {e}")