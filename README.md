# House Price Prediction Using Machine Learning
## Candidate

Ashmita Thiyam
## Domain

Data Science
## Project Overview

This project focuses on predicting house prices using machine learning techniques. The Ames Housing dataset was used to analyse the relationship between different property characteristics and house prices.

Three regression models were implemented and compared:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression

Random Forest Regression achieved the best overall performance and was integrated into an interactive Streamlit dashboard.



## Dataset

**Dataset:** Ames Housing Dataset  
**Source:** Kaggle  
**Number of Properties:** 2,930  
**Target Variable:** SalePrice

The dataset contains property-related features such as living area, overall quality, lot area, year built, bedrooms, bathrooms, garage information, neighborhood and other characteristics.

## Machine Learning Models

### 1. Linear Regression

Used as a baseline regression model.

### 2. Decision Tree Regression

Used to capture non-linear relationships between property characteristics and house prices.

### 3. Random Forest Regression

Used as the final model because it achieved the best performance among the evaluated models.

## Model Results

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.887962 | $17,364.58 | $29,971.12 |
| Decision Tree Regression | 0.814346 | $25,049.93 | $38,580.96 |
| **Random Forest Regression** | **0.923131** | **$15,260.49** | **$24,825.36** |

The Random Forest model achieved an R² score of **0.9231**, meaning it explains approximately 92.31% of the variance in house prices in the test set.

## Feature Engineering

The project includes the following engineered features:

- `TotalSF`
- `TotalBathrooms`
- `TotalPorchSF`

These features combine related property measurements to provide additional information to the regression models.

## Dashboard

The project includes an interactive Streamlit dashboard that allows users to:

- Enter property characteristics
- Generate a house-price prediction
- View dataset statistics
- Explore house-price distributions
- Analyse relationships between property features and price
- Compare machine learning models
- View Random Forest feature importance
- Compare actual and predicted prices

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
- Jupyter Notebook

## Project Files

```text
house_pred/
│
├── app.py
├── House_Price_Prediction.ipynb
├── cleaned_AmesHousing.csv
├── house_price_model.pkl
├── requirements.txt
└── README.md
