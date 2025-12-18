

CNN-Based Image Similarity System Using ResNet:
Overview:

This project implements a content-based image similarity system using deep learning. Given a query image, the system retrieves the Top-K visually similar images from an image database by comparing convolutional neural network (CNN) feature embeddings. The project focuses on representation learning and similarity search rather than image classification.

Core Idea:

Instead of comparing raw pixel values, each image is mapped to a fixed-length feature vector (embedding) using a pretrained ResNet model. Images that share similar visual characteristics produce embeddings that are closer in the feature space, enabling effective similarity-based retrieval.

Methodology:
Feature Extraction:

A pretrained ResNet-18 model from torchvision is used as the backbone. The final classification layer is removed to obtain 512-dimensional feature embeddings. Transfer learning is leveraged to reuse visual representations learned from the ImageNet dataset.

Image Preprocessing:

All images are resized to 224 × 224 pixels and normalized using ImageNet mean and standard deviation values. This ensures consistency with the pretrained model’s training distribution and improves embedding quality.

Similarity Computation:

Cosine similarity is used to compare embeddings between the query image and the database images. The Top-K images with the highest similarity scores are retrieved.

Evaluation:

Since labeled similarity data is not available, qualitative evaluation is performed by visualizing the query image alongside the retrieved results with similarity scores.


How to Run
Dependencies

Install the required libraries:

pip install torch torchvision numpy matplotlib opencv-python scikit-learn

Execution (Kaggle / Jupyter)

To ensure visualization works correctly, run the pipeline inside the notebook kernel:

import main_multi
main_multi.main()


This will build the embedding database, select a query image, retrieve the Top-K similar images, and display the results.

Output

The system displays the query image followed by the Top-K retrieved images along with their cosine similarity scores. This provides an intuitive and interpretable validation of the similarity pipeline.

Technologies Used

Python, PyTorch, Torchvision, NumPy, OpenCV, Matplotlib, scikit-learn
