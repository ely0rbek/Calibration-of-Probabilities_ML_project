# Assignment 2 — Calibration of Probabilities

This project is based on the paper *Predicting Good Probabilities with Supervised Learning* by Alexandru Niculescu-Mizil and Rich Caruana (ICML 2005). The main focus of the project is to evaluate and improve the calibration of probabilistic classifiers using real-world medical datasets.

The project investigates how machine learning models can produce reliable probability estimates in addition to accurate predictions. Special attention is given to calibration quality, confidence estimation, and reliability analysis for healthcare-related classification tasks.

## Project Goals

- Evaluate the calibration of probabilistic classifiers
- Analyze prediction confidence and reliability
- Compare classification performance and calibration quality
- Apply calibration analysis to medical datasets
- Generate reliability diagrams and calibration metrics
- Study the relationship between prediction accuracy and probability estimation

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

