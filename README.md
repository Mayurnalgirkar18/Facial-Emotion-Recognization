# 🎭 Facial Emotion Recognition using Deep Learning (CNN)
# Overview
This project focuses on Facial Emotion Recognition (FER) using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.
The model classifies grayscale facial images into multiple emotion categories and demonstrates a complete end-to-end deep learning workflow, from data preprocessing to model evaluation and saving.
This project is designed to showcase practical deep learning skills relevant for Machine Learning / Deep Learning.

 # Key Objectives
Build a robust CNN for image-based emotion classification
Apply best practices in data preprocessing and regularization
Evaluate model performance using industry-standard metrics
Create a reusable and deployable trained model

# Skills Demonstrated

Convolutional Neural Networks (CNN)
Image preprocessing & normalization
Data augmentation techniques
Model regularization (Dropout, Batch Normalization)
Multi-class classification
Performance evaluation (Confusion Matrix, Classification Report)
TensorFlow / Keras model saving and reuse

# Dataset

Image size: 48 × 48
Color mode: Grayscale
Task: Multi-class facial emotion classification
Common FER dataset structure (train/test folders)
Each image represents a human face labeled with an emotion such as:

1.Angry
2.Happy
3.Sad
4.Fear
5.Surprise
6.Neutral
7.Disgust

# Model Architecture (High-Level)

* Multiple Conv2D + ReLU layers
* Ma+xPooling layers for feature reduction
* Batch Normalization to stabilize training
* Dropout to reduce overfitting
* Fully connected (Dense) layers
* Softmax output layer for emotion prediction
* The architecture balances performance and computational efficiency, making it suitable for both local GPU and cloud environments.

# Project Pipeline

* Data loading from directory structure
* Image preprocessing & normalization
* Data augmentation for generalization
* CNN model design
* Model compilation and training
* Performance evaluation
* Visualization of results
* Model saving (.h5 format)

# Results & Evaluation

* Training and validation accuracy tracked across epochs
* Model evaluated using:
* Confusion Matrix
* Precision, Recall, and F1-Score
* Helps identify class-wise strengths and weaknesses
* The evaluation demonstrates understanding beyond accuracy, focusing on real-world model behavior.

# Tech Stack
* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
