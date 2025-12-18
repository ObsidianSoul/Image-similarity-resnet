import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from model_util import load_model, extract_embedding
from image_util_cli import show_results

BASE_DIR = "/kaggle/working/image-similarity-resnet"
TRAIN_DIR = f"{BASE_DIR}/data/train"
TEST_DIR  = f"{BASE_DIR}/data/test"
TOP_K = 5

def main():
    model = load_model()

    embeddings = []
    image_paths = []

    print("Building embedding database...")

    for root, _, files in os.walk(TRAIN_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, file)
                emb = extract_embedding(model, img_path)
                embeddings.append(emb)
                image_paths.append(img_path)

    embeddings = np.array(embeddings)
    print(f"Total images indexed: {len(image_paths)}")

    query_image = os.path.join(TEST_DIR, os.listdir(TEST_DIR)[0])
    query_emb = extract_embedding(model, query_image)

    sims = cosine_similarity(query_emb.reshape(1, -1), embeddings)[0]
    top_idx = sims.argsort()[-TOP_K:][::-1]

    query_img = cv2.imread(query_image)
    result_imgs = [cv2.imread(image_paths[i]) for i in top_idx]
    scores = [float(sims[i]) for i in top_idx]

    print("\nQuery image:")
    print(query_image)

    print("\nTop-K results:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank}: {image_paths[idx]}  |  score = {sims[idx]:.4f}")

    show_results(query_img, result_imgs, scores)

if __name__ == "__main__":
    main()
