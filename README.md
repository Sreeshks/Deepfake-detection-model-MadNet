# Enhanced MADNet - Multi-Class Deepfake Detection

## Overview
Enhanced MADNet is an advanced deepfake detection system that classifies videos into multiple categories of manipulation artifacts rather than just binary real/fake classification. It combines spatial and temporal analysis using EfficientNetB0 and LSTM networks.

## Features
- Multi-class classification for different types of deepfake artifacts:
  - Real content
  - Face manipulation
  - Expression manipulation
  - Lighting artifacts
  - Blending inconsistencies
- Enhanced feature extraction with multi-scale spatial and temporal analysis
- Improved architecture with additional LSTM and Dense layers
- Categorical cross-entropy loss for multi-class prediction

## Requirements
- Python 3.8+
- TensorFlow 2.5+
- NumPy
- OS standard library

## Installation
```bash
pip install tensorflow numpy


data/
├── train/
│   ├── real/
│   ├── deepfake_face/
│   ├── deepfake_expression/
│   ├── deepfake_lighting/
│   └── deepfake_blending/
├── valid/
│   └── [same subdirectories]
└── test/
    └── [same subdirectories]

  Usage
Prepare your dataset in the required structure
Update the data paths in the example usage section
Run the script:
python madnet_enhanced.py
**Model Architecture**
Base Model: EfficientNetB0
Temporal Analysis: Dual LSTM layers (256 and 128 units)
Feature Combination: Multi-scale spatial and temporal features
Classification: Softmax output for 5 classes
Regularization: Dropout (0.5, 0.4, 0.3) and Batch Normalization
**Training Parameters**
Optimizer: Adam (learning_rate=0.0001)
Loss: Categorical Cross-entropy
Metrics: Accuracy, Multi-label AUC
Callbacks: EarlyStopping, ReduceLROnPlateau
Improvements over Original MADNet
Multi-class classification instead of binary
Enhanced feature extraction with dual LSTM layers
Deeper classification head with additional dense layers
Multi-scale feature analysis
More robust regularization
**Contributing**
Feel free to submit pull requests or issues for improvements or bug fixes.


**License**
MIT License


Key improvements in the code:
1. Changed from binary to multi-class classification (5 classes)
2. Added multiple artifact types in the dataset preparation
3. Used softmax instead of sigmoid for multi-class output
4. Switched to categorical_crossentropy loss
5. Added multi-scale feature extraction with dual LSTM layers
6. Enhanced the dense layers structure for better classification
7. Added one-hot encoding for labels
8. Improved metrics with multi-label AUC

The code maintains the original functionality while adding the ability to detect and classify different types of deepfake artifacts, making it more practical for real-world applications where identifying the specific type of manipulation is valuable.
