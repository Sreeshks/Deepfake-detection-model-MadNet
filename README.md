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