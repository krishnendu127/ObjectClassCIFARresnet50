# Object Recognition with CIFAR-10 Dataset

## Overview
This project aims to recognize objects in images using machine learning techniques. It utilizes the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to build a neural network model that accurately classifies these images into their respective categories.

## Dataset
The CIFAR-10 dataset is used for this project, which is available through the Kaggle competition platform. It contains labeled images across 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Features
- **Data Download and Extraction**: The Kaggle API is used to download the CIFAR-10 dataset, which is then extracted and processed for training.
- **Labels Processing**: The labels for the images are processed from a CSV file and converted into numerical representations for classification.
- **Image Processing**: Images are loaded, resized, and converted into numpy arrays for further processing.
- **Data Scaling**: The pixel values of the images are scaled to a range of [0, 1] to normalize the data.
- **Neural Network Architecture**: Two different neural network architectures are explored: a custom model with dense layers and a pre-trained ResNet50 model fine-tuned for object recognition.
- **Model Training**: The neural network models are trained on the training data, with a validation split to monitor performance during training.
- **Model Evaluation**: The trained models are evaluated on the test data to assess their classification accuracy.

## Dependencies
- NumPy
- Pandas
- PIL (Python Imaging Library)
- Matplotlib
- OpenCV (cv2)
- TensorFlow
- TensorFlow Hub
- Kaggle
- py7zr (for handling 7z archives)

## Usage
1. Ensure you have the necessary dependencies installed, including the Kaggle API key for dataset download.
2. Run the provided code in a Python environment such as Jupyter Notebook or Google Colab.
3. The code will download, preprocess, train, evaluate, and make predictions using the CIFAR-10 dataset.
4. You can modify the code, experiment with different neural network architectures or hyperparameters, and explore further improvements in classification accuracy.
