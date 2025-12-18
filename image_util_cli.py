import matplotlib.pyplot as plt
import cv2

def show_results(query_img, retrieved_imgs, scores):
    plt.figure(figsize=(15,4))

    plt.subplot(1, len(retrieved_imgs)+1, 1)
    plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    plt.title("Query")
    plt.axis("off")

    for i, (img, score) in enumerate(zip(retrieved_imgs, scores)):
        plt.subplot(1, len(retrieved_imgs)+1, i+2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{score:.2f}")
        plt.axis("off")

    plt.show()
