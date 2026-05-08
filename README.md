# Medical Data Analysis and Calibration

This project focuses on machine learning analysis of medical datasets, including breast cancer and heart disease prediction tasks. The main objective is to evaluate model performance, probability calibration, and reliability of predictions on real-world healthcare data.

## Project Goals

- Load and preprocess medical datasets
- Train machine learning classification models
- Evaluate prediction performance
- Analyze calibration quality and reliability
- Generate visualizations and performance metrics

## Implemented Features

- Breast Cancer dataset processing
- Heart Disease dataset preprocessing
- Binary target encoding
- Missing value handling
- Train/test splitting
- Model evaluation metrics
- Reliability diagrams
- Expected Calibration Error (ECE) analysis
- Visualization generation

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Dataset Information

### Breast Cancer Dataset
The breast cancer dataset uses:
- Malignant (M) → 1
- Benign (B) → 0

### Heart Disease Dataset

The Heart Disease dataset consists of multiple subsets collected from different locations, including Cleveland, Hungary, Switzerland, and Long Beach VA.

In this project, only the `processed.cleveland.data` subset is used because:
- it is the most complete and well-structured subset,
- contains fewer missing values,
- is widely used as a benchmark in research,
- and is already preprocessed for machine learning tasks.

## How to Run

- 1. Clone the Repository
- 2. Run full_analysis.ipynb or full_analysis.py file.


## Project Structure

``` id="9owm4j"
project/
│
├── Datasets/
├── figures/
├── Report/
├── full_analysis.py
└── full_analysis.ipynb

