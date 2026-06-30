import requests
import json
import os

# Configuration
img_path = "/home/fusion-gpu/fusion-projects/face-recognition-server/downloaded_images/employee-emerald-e00643/employee-emerald-e00643-2.jpg"
api_url = "http://localhost:8001/v1/faces/recognize_upload"
api_key = "fs_9f2b8a71c4d04e5e9b3d8a7c6b5a4f3e"
branch = "live-emp-TMJ-CBE"

print(f"Testing recognition for: {img_path}")
print(f"Target Branch: {branch}")

if not os.path.exists(img_path):
    print(f"ERROR: Image not found at {img_path}")
    exit(1)

# Prepare multipart form data
files = {
    'file': ('image.jpg', open(img_path, 'rb'), 'image/jpeg')
}
data = {
    'branch': branch,
    'top_k': 5,
    'min_similarity': 0.1  # Low threshold to ensure we get results for testing
}
headers = {
    'x-api-key': api_key
}

try:
    response = requests.post(api_url, headers=headers, files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    
    if response.status_code == 200:
        print("\nSUCCESS: API call complete. Check your logs for 'Inserted recognition event'.")
    else:
        print(f"\nFAILED: {response.text}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    files['file'][1].close()
