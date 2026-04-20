import httpx
import base64
import json
import os

async def test_quality_check():
    url = "http://localhost:8001/v1/quality/check_upload"
    img_path = "/mnt/additional-disk/face_service/ui/src/assets/BRTMJ-CBE__ZNFirstFloor__IOENTRY__CAMFF-Lift-Entry_10.122.15.59_01_20260411184026985_FACE_BACKGROUND_32180.webp"
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(img_path, "rb") as f:
            files = {"file": ("image.webp", f, "image/webp")}
            print(f"Sending request to {url}...")
            response = await client.post(url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            num_faces = len(result.get("faces", []))
            print(f"Success! Detected {num_faces} faces.")
            print(json.dumps(result, indent=2)[:1000] + "...") # Truncate for display
            
            if "annotated_image" in result:
                print("\nAnnotated image received (Base64 string present)")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_quality_check())
