import cv2
import numpy as np
import os
import sys
from embedders.buffalo_l import BuffaloLEmbedder

def compare_two_images(img_path1, img_path2):
    model_root = os.environ.get("BUFFALO_MODEL_ROOT", "/home/fusion-gpu/fusion-projects/face-recognition-server/buffalo_l/models")
    model_name = os.environ.get("BUFFALO_MODEL_NAME", "buffalo_l")
    
    print(f"Loading model {model_name} from {model_root}...")
    embedder = BuffaloLEmbedder(model_root=model_root, model_name=model_name)
    
    # Load images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None:
        print(f"Error: Could not read image at {img_path1}")
        return
    if img2 is None:
        print(f"Error: Could not read image at {img_path2}")
        return

    try:
        print(f"Extracting embedding for {img_path1}...")
        emb1 = embedder.embed_bgr(img1)
        
        print(f"Extracting embedding for {img_path2}...")
        emb2 = embedder.embed_bgr(img2)
        
        # Cosine similarity (dot product of normalized vectors)
        similarity = np.dot(emb1, emb2)
        
        print("\n" + "="*30)
        print(f"Result: {similarity:.4f}")
        print("="*30)
        
        if similarity > 0.45:
            print("Conclusion: Match (Highly likely same person)")
        elif similarity > 0.35:
            print("Conclusion: Potential Match (Uncertain)")
        else:
            print("Conclusion: No Match")
            
    except Exception as e:
        print(f"Error during comparison: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 compare_images.py <path_to_img1> <path_to_img2>")
    else:
        compare_two_images(sys.argv[1], sys.argv[2])
