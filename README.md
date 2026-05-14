# Retail Sales Forecasting

## Overview

This project forecasts weekly retail sales using historical store-level sales data. The objective is to compare baseline and advanced machine learning models for predicting weekly demand and identifying the most effective forecasting approach.

The project demonstrates an end-to-end ML workflow including:

* data preprocessing
* feature engineering
* model training
* evaluation
* prediction visualization

---

## Problem Statement

Retail businesses rely on accurate sales forecasting to optimize:

* inventory planning
* staffing
* supply chain operations
* promotional strategies

This project predicts weekly sales based on:

* store information
* department information
* holiday indicators
* historical trends

---

## Models Used

### 1. Linear Regression

Used as a baseline model to establish benchmark forecasting performance.

### 2. XGBoost Regressor

Used as an advanced ensemble model to capture non-linear relationships and improve forecasting accuracy.

---

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* matplotlib
* seaborn
* YAML configuration

---

## Project Structure

```bash
Retail-Sales-Forecasting/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── visualization.py
│
├── config.yaml
├── main.py
└── README.md
```

---

## Workflow

### 1. Data Loading

The pipeline loads retail sales data from CSV format.
If the dataset is unavailable, dummy sample data is automatically generated for testing.

### 2. Data Preprocessing

The preprocessing module:

* cleans data
* converts date fields
* prepares model-ready features
* handles categorical variables

### 3. Train/Test Split

The dataset is split into training and testing sets using configurable parameters from `config.yaml`.

### 4. Model Training

Two forecasting models are trained:

* Linear Regression
* XGBoost Regressor

### 5. Model Evaluation

Models are evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score

### 6. Visualization

Prediction outputs are visualized to compare actual vs predicted sales performance.

---

## Example Output

```python
Linear Regression Results
MAE: XXXX
RMSE: XXXX
R2 Score: XXXX

XGBoost Results
MAE: XXXX
RMSE: XXXX
R2 Score: XXXX
```

---

## Key Learnings

* Comparing baseline vs ensemble forecasting models
* Building modular ML pipelines
* Using configuration-driven workflows
* Evaluating forecasting performance using regression metrics
* Structuring reusable ML project architecture

---

## Future Improvements

* Add time-series specific models (ARIMA / Prophet)
* Add hyperparameter tuning
* Deploy as API using FastAPI
* Add experiment tracking using MLflow
* Add drift monitoring for production forecasting systems

---

## Business Impact

Accurate retail demand forecasting can help businesses:

* reduce inventory waste
* improve stock availability
* optimize operational planning
* support data-driven decision-making

---

## Author

Shruti Bhosale
