## Overview

This repository implements a **Python-based Celebrity Face Recognition** pipeline that:

- **Automatically detects and crops faces** using OpenCV’s Haar cascades  
- **Extracts multi-level wavelet-transform features** for enhanced feature representation  
- **Stacks raw pixel data with wavelet features** to form a robust feature vector  
- **Trains and compares multiple classifiers** (SVM, Random Forest, Logistic Regression) via grid search  
- **Exports the best-performing Logistic Regression model** along with a class-to-index mapping

## Key Features

- **Face & Eye Detection**  
  Utilizes OpenCV’s `haarcascade_frontalface_default.xml` and `haarcascade_eye_tree_eyeglasses.xml` to ensure only clear face regions (with at least two eyes) are processed.

- **Wavelet Transform Preprocessing**  
  Applies PyWavelets (`pywt`) to decompose each face image, zero out approximation coefficients, then reconstruct and normalize the detail component for richer texture features.

- **Pipeline & Hyperparameter Tuning**  
  - Scales features with `MinMaxScaler`  
  - Uses `GridSearchCV` for exhaustive hyperparameter tuning across SVM, Random Forest, and Logistic Regression  
  - Reports best validation scores and optimal parameters for each algorithm

- **Model Persistence**  
  - Saves the class-label mapping in `class_dir.json`  
  - Exports the final Logistic Regression model to `celeb_image_clf_LG.pkl` via `joblib`

## Prerequisites

- Python 3.7+  
- OpenCV (`cv2`)  
- PyWavelets (`pywt`)  
- scikit-learn  
- pandas, NumPy, matplotlib (for optional analysis/plots)  

Install dependencies:
```bash
pip install opencv-python pywt scikit-learn pandas matplotlib

```
AI generated README.md file
