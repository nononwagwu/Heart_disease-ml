# Heart Disease Prediction with Machine Learning

A machine learning project exploring the use of classification models to predict the presence of heart disease from patient health and clinical attributes.

## Overview

This project applies supervised machine learning techniques to heart disease datasets to explore how clinical features can be used to classify heart disease risk.

The project includes data preprocessing, exploratory analysis, model training, hyperparameter tuning, and model evaluation.

> This project is intended for educational and research purposes and is not a medical diagnostic tool.

## Dataset

The project uses heart disease data from the UCI Machine Learning Repository, including data collected from multiple sources such as:

- Cleveland
- Hungarian
- Switzerland
- Long Beach VA

The datasets contain clinical attributes related to cardiovascular health, including features such as age, cholesterol levels, resting blood pressure, maximum heart rate, and other patient measurements.

## Machine Learning Pipeline

The project follows a machine learning workflow that includes:

- Loading and combining heart disease datasets
- Cleaning and preprocessing clinical data
- Handling missing values
- Preparing features for model training
- Training classification models
- Performing model tuning
- Evaluating model performance
- Visualizing model results

## Models

The project explores classification approaches including:

- Logistic Regression
- Additional segmented and tuned model configurations

Further model comparisons can be added as the project develops.

## Tech Stack

- **Python**
- **pandas** — data manipulation and preprocessing
- **NumPy** — numerical computing
- **scikit-learn** — machine learning and model evaluation
- **Matplotlib** — data visualization

## Project Structure

```text
Heart_disease-ml/
├── heart_logistic.py      # Main machine learning pipeline
├── costs/                 # Supporting project files
├── cleveland.data         # Cleveland dataset
├── hungarian.data         # Hungarian dataset
├── long-beach-va.data     # Long Beach VA dataset
└── README.md
