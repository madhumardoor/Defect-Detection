# Defect Detection using CNN with Synthetic Augmentation
## Project Overview
This project aims to detect manufacturing defects using a Convolutional Neural Network (CNN) trained on the MVTec Anomaly Detection Dataset (Bottle category). It includes data augmentation techniques to generate synthetic defects and improve model performance.

## Dataset
Source: MVTec AD Dataset

Category: bottle

Images:

Normal: 209

Defective: 83

## Technologies Used
Python

PyTorch

OpenCV

Albumentations

Matplotlib

scikit-learn

## Workflow
Data Loading
Load and visualize normal and defective samples using glob and OpenCV.

Custom Dataset Class
Created a PyTorch Dataset to load images with labels (0: normal, 1: defect).

CNN Model
Built a simple CNN classifier with three convolutional layers.

Training
Trained the model for 5 epochs using cross-entropy loss and Adam optimizer.

Synthetic Data Generation
Used Albumentations to apply transformations (noise, contrast, elastic) and generate synthetic defects.

Retraining
Model was retrained with the augmented dataset to improve accuracy.

## Evaluation

Initial Accuracy: ~77%

Final Accuracy After Augmentation: 83.05%

 ## Results
Phase	Accuracy
Before Augmentation	~77%
After Augmentation	83.05%

## Sample Visualizations
Sample input images (normal and defect)

Synthetic augmentations applied

Loss graph (if plotted)

## Future Improvements
Test with other MVTec categories (e.g., metal nut, hazelnut)

Implement Grad-CAM for visual explainability

Deploy model with a Flask API for real-time prediction

## Author
Madhu M 
