# IPL Match Outcome Predictor

This project implements a machine learning system to predict the outcome of Indian Premier League (IPL) matches using structured historical match data. The trained model is integrated into an interactive Streamlit web application for real-time prediction.

## Overview

The objective of this project is to design a reproducible and user-facing machine learning pipeline capable of generating match outcome predictions based on pre-match contextual features.

The system includes data preprocessing, feature engineering, model training, evaluation, and application deployment.

## Dataset

- Historical IPL match dataset (Kaggle)
- Match-level structured data
- Cleaned and preprocessed to handle missing values and inconsistencies

## Feature Engineering

Key features include:

- Team performance metrics  
- Venue influence  
- Toss result  
- Recent team form  
- Head-to-head statistics  

Categorical variables were encoded appropriately before model training.

## Model Development

- Implemented Logistic Regression for binary classification on structured tabular data.
- Performed train-test split (80-20).
- Applied cross-validation to improve generalization.
- Evaluated model using confusion matrix and classification metrics.
- Analyzed feature coefficients to interpret model behavior.
- Serialized trained model for integration into the application.

Logistic Regression was selected due to its interpretability and strong performance on structured datasets.

## Deployment

The trained model is deployed using Streamlit.

- Interactive user interface for entering match details  
- Real-time prediction output  
- Model loaded from serialized file  
- Designed for demonstration and experimentation  

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib / Seaborn  

## Key Learnings

- End-to-end ML pipeline design  
- Importance of structured feature engineering  
- Preventing data leakage  
- Model interpretability using coefficient analysis  
- Deploying ML models into interactive applications  

## Future Improvements

- Compare with ensemble models (Random Forest, Gradient Boosting)
- Add probability score output
- Deploy on cloud infrastructure
- Improve UI and scalability

## Live Application

The application is publicly accessible:

Live Demo: https://predict-the-ipl-5.streamlit.app/
## Author

Pranshul Khokhar  
AI/ML & Software Engineer
