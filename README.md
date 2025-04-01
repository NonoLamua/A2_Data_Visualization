# A2_Data_Visualization

# ML Model Trainer

A streamlined Streamlit application for training and evaluating machine learning models without coding.

# Live Demo

You can access the live application here: https://a2datavisualizationantoniolamua.streamlit.app/

# Overview

This application provides an intuitive interface for:

Loading datasets (built-in or custom)
Selecting features and target variables
Configuring and training ML models
Visualizing model performance metrics
Exporting trained models for later use

# Features
# Dataset Management

Use pre-loaded datasets from Seaborn (iris, titanic, diamonds, tips, planets)
Upload custom CSV files
View dataset previews and summaries

# Feature Selection

Choose target variables
Select numerical and categorical features
Automatic problem type detection (classification/regression)

# Model Configuration

Choose appropriate algorithms based on problem type:

Regression: Linear Regression, Random Forest Regressor
Classification: Logistic Regression, Random Forest Classifier
Configure model hyperparameters
Set test/train split ratio

# Results Visualization

Performance metrics:

Classification: Accuracy, confusion matrix, classification report, ROC curve
Regression: MSE, R², residual analysis


Feature importance visualization
Interactive charts and plots

# Model Export

Download trained models as pickle files for production use

# Technical Implementation

Automatic preprocessing of data (handling categorical variables, missing values)
Feature engineering with label encoding and one-hot encoding
Cross-validation and model evaluation
Interactive visualization using Matplotlib and Seaborn
Streamlit components for responsive UI

# requirements.txt
Copiarstreamlit
pandas
numpy
seaborn
matplotlib
scikit-learn
pickle-mixin
