# Enhanced MADNet - Multi-Class Deepfake Detection

## Overview
Enhanced MADNet is an advanced deepfake detection system that classifies videos into multiple categories of manipulation artifacts rather than just a binary real/fake classification. It combines spatial and temporal analysis using EfficientNetB0 and LSTM networks.

## Features
- **Multi-class classification** for different types of deepfake artifacts:
  - Real content
  - Face manipulation
  - Expression manipulation
  - Lighting artifacts
  - Blending inconsistencies
- **Enhanced feature extraction** with multi-scale spatial and temporal analysis
- **Improved architecture** with additional LSTM and Dense layers
- **Categorical cross-entropy loss** for multi-class prediction

## Requirements
- Python 3.8+
- TensorFlow 2.5+
- NumPy
- OS standard library

## Installation
```bash
pip install tensorflow numpy
```

## Dataset Structure
```
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
```

## Dataset Links
Here are some publicly available datasets commonly used for deepfake detection:

1. **FaceForensics++**
   - **Description**: A widely-used dataset with over 1,000 original video sequences manipulated using various deepfake techniques (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
   - **Link**: [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics)
   - **Note**: You may need to request access from the Technical University of Munich team.

2. **Celeb-DF**
   - **Description**: A large-scale dataset with 590 real videos from YouTube and 5,639 deepfake videos.
   - **Link**: [Celeb-DF GitHub](https://github.com/danmohaha/celeb-df)
   - **Note**: Version 2 (Celeb-DF v2) is the most comprehensive.

3. **DeepFake Detection Challenge (DFDC) Dataset**
   - **Description**: One of the largest datasets with over 100,000 videos, created by Facebook and partners.
   - **Link**: [DFDC Website](https://ai.meta.com/datasets/dfdc/)
   - **Note**: Requires registration to download.

## Usage
1. Prepare your dataset in the required structure.
2. Update the data paths in the example usage section.
3. Run the script:
```bash
python madnet_enhanced.py
```

## Model Architecture
- **Base Model:** EfficientNetB0
- **Temporal Analysis:** Dual LSTM layers (256 and 128 units)
- **Feature Combination:** Multi-scale spatial and temporal features
- **Classification:** Softmax output for 5 classes
- **Regularization:** Dropout (0.5, 0.4, 0.3) and Batch Normalization

## Training Parameters
- **Optimizer:** Adam (learning_rate=0.0001)
- **Loss:** Categorical Cross-entropy
- **Metrics:** Accuracy, Multi-label AUC
- **Callbacks:** EarlyStopping, ReduceLROnPlateau

## Improvements Over Original MADNet
1. Multi-class classification instead of binary.
2. Added multiple artifact types in the dataset preparation.
3. Used softmax instead of sigmoid for multi-class output.
4. Switched to categorical_crossentropy loss.
5. Added multi-scale feature extraction with dual LSTM layers.
6. Enhanced the dense layers structure for better classification.
7. Added one-hot encoding for labels.
8. Improved metrics with multi-label AUC.

## Contributing
Feel free to submit pull requests or issues for improvements or bug fixes.

## License
MIT License

## Summary
The code maintains the original functionality while adding the ability to detect and classify different types of deepfake artifacts. This makes it more practical for real-world applications where identifying the specific type of manipulation is valuable.

