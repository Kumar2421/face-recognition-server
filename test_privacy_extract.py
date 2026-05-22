import requests
import base64
import json
import os

# Use a real image from the project if possible, or a placeholder
image_path = "privacy_check_face_0.jpg"
if not os.path.exists(image_path):
    # Fallback to downloading a sample if not found
    print(f"{image_path} not found. Testing with a remote image.")
    image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    resp = requests.get(image_url)
    image_b64 = base64.b64encode(resp.content).decode('utf-8')
else:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')

url = "http://localhost:8001/v1/faces/privacy_extract"
payload = {"image_b64": image_b64}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Number of faces extracted: {len(data.get('results', []))}")
        for i, result in enumerate(data.get('results', [])):
            print(f"Face {i}: Quality keys: {list(result.get('quality', {}).keys()) if result.get('quality') else 'None'}")
            # Save the first result to verify visually if needed
            if i == 0:
                img_data = result['image_b64'].split(",")[1]
                with open(f"privacy_extract_output_{i}.jpg", "wb") as f:
                    f.write(base64.b64decode(img_data))
                print(f"Saved Face {i} to privacy_extract_output_{i}.jpg")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Failed to connect to server: {e}")
