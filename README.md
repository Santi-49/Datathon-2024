# Real Estate Price Prediction Model

## Project Overview

This project implements an advanced machine learning pipeline for predicting residential property prices in the city of Chicago, Illinois, using a comprehensive dataset of real estate listings. The solution leverages CatBoost regression with sophisticated feature engineering and model interpretability techniques to deliver accurate property valuations.

The project demonstrates end-to-end data science capabilities including data preprocessing, feature engineering, model training with hyperparameter optimization, and model interpretation using SHAP (SHapley Additive exPlanations) values.

## Business Problem

Predicting accurate real estate prices is critical for buyers, sellers, and financial institutions. This project addresses the challenge of forecasting property closing prices based on historical transaction data, property characteristics, and location features across nearly a year of market activity in Illinois.

## Technical Approach

### Model Selection: CatBoost Regressor

CatBoost (Categorical Boosting) was selected as the primary modeling framework for several key technical reasons:

**Native Categorical Handling**: CatBoost processes categorical features directly without requiring manual encoding transformations, reducing preprocessing complexity and potential information loss.

**Gradient Boosting Excellence**: Developed by Yandex, CatBoost implements state-of-the-art gradient boosting algorithms optimized for both numerical and categorical data, consistently achieving superior performance on structured data tasks.

**Regularization and Overfitting Prevention**: Built-in ordered boosting and symmetric tree structures provide robust protection against overfitting, particularly valuable when working with high-cardinality categorical features.

**Computational Efficiency**: The framework supports both CPU and GPU execution, enabling scalable training on large datasets while maintaining production-ready inference speeds.

### Data Engineering Pipeline

The data preprocessing and feature engineering pipeline implements several critical steps:

**Outlier Detection and Removal**: Statistical analysis identified and removed extreme outliers in the target variable (properties exceeding $13 million) to improve model robustness and prevent skewed predictions.

**Feature Relevance Analysis**: Systematic evaluation of feature importance eliminated highly granular columns with limited predictive power, including specific street addresses, transaction dates, and administrative identifiers. This dimensionality reduction enhanced model efficiency without sacrificing accuracy.

**Categorical Feature Optimization**: Analysis of unique value distributions in categorical columns identified features with excessive cardinality that could introduce noise. Features were filtered based on a 50% uniqueness threshold to maintain signal-to-noise ratio.

**List-Type Feature Unpacking**: Complex features stored as list structures (e.g., lot features, basement types, cooling systems, parking features) were systematically unpacked into binary indicator variables through custom one-hot encoding logic, preserving the granularity of multi-valued attributes.

**Missing Data Strategy**: Missing values in categorical features were explicitly encoded as 'nan' strings rather than dropped, allowing the model to learn patterns associated with missing information.

**Train-Test Consistency**: Rigorous alignment procedures ensured that test set features matched the training set schema, with missing columns initialized to zero to prevent prediction failures.

### Model Training and Optimization

The training process employed a validation-based approach to optimize model performance:

**Hyperparameter Tuning**: Systematic exploration of key parameters including learning rate (0.05), L2 regularization (18), minimum data in leaf (150), and feature subsampling (40%) to balance bias-variance tradeoff.

**Early Stopping**: Implementation of early stopping with 100-round patience on validation sets to identify optimal iteration counts (7,247 iterations) and prevent overfitting.

**Loss Function**: Mean Absolute Error (MAE) was selected as the primary optimization objective, providing robust performance against outliers while maintaining interpretability for stakeholders.

**Random Seed Control**: Consistent random seeds across train-test splits and model initialization ensure reproducibility of results.

### Model Interpretability

SHAP (SHapley Additive exPlanations) analysis provides transparent insights into model decision-making:

**Global Feature Importance**: Summary plots reveal which features drive predictions across the entire dataset, enabling feature selection and domain understanding.

**Local Explanations**: Waterfall plots for individual predictions decompose the contribution of each feature to specific property valuations, supporting explainability requirements.

**Expected Value Baseline**: SHAP values are calculated relative to the model's expected prediction, quantifying both positive and negative feature impacts.

## Project Architecture

```
main.py              - End-to-end pipeline orchestration
data_module.py       - Data I/O and model persistence utilities
shap_module.py       - SHAP explainer implementation and visualization
requirements.txt     - Python dependency specifications
data/
  train.csv          - Historical property transaction data
  test.csv           - Properties requiring price predictions
  train_fe.csv       - Engineered training features
  test_fe.csv        - Engineered test features
  categorical.csv    - Categorical feature index
  model_catboost.sav - Serialized trained model
submission.csv       - Final price predictions
```

## Key Technical Components

**main.py**: Executes the complete machine learning workflow including data loading, feature engineering, model training with hyperparameter optimization, SHAP value generation, and prediction export.

**data_module.py**: Provides abstraction layer for data serialization and deserialization using Pandas CSV operations and joblib model persistence, ensuring consistent data handling across pipeline stages.

**shap_module.py**: Implements SHAP TreeExplainer for gradient boosting models, generating both global summary plots and local waterfall explanations for model interpretation.

## Technologies and Libraries

**Core ML Framework**: CatBoost 1.2+ for gradient boosted decision trees  
**Data Processing**: Pandas 1.5.3, NumPy for efficient dataframe operations  
**Model Evaluation**: Scikit-learn for train-test splitting and metrics  
**Model Interpretation**: SHAP for explainable AI  
**Visualization**: Matplotlib for SHAP plot rendering  
**Model Persistence**: Joblib for efficient model serialization

## Installation and Setup

**Clone the Repository**:
```bash
git clone https://github.com/Santi-49/Datathon-2024.git
cd Datathon-2024
```

**Create Virtual Environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install Dependencies**:
```bash
pip install -r requirements.txt
```

## Execution

Run the complete pipeline:
```bash
python main.py
```

The pipeline executes the following operations:
1. Loads raw training and test datasets from `./data/`
2. Performs feature engineering including outlier removal, feature unpacking, and categorical encoding
3. Exports processed datasets to `train_fe.csv` and `test_fe.csv`
4. Trains CatBoost regressor with optimized hyperparameters
5. Serializes trained model to `./data/model_catboost.sav`
6. Generates SHAP interpretability plots
7. Produces final predictions in `submission.csv`

## Results and Performance

The optimized CatBoost model achieved strong predictive performance through:
- Effective handling of 200+ engineered features including high-cardinality categorical variables
- Optimal convergence at 7,247 boosting iterations with MAE-based optimization
- Robust generalization through L2 regularization and feature subsampling
- Transparent predictions validated through SHAP analysis

## Skills Demonstrated

**Machine Learning**: Gradient boosting, hyperparameter optimization, cross-validation, ensemble methods  
**Data Engineering**: Feature engineering, one-hot encoding, data cleaning, outlier detection  
**Model Interpretation**: SHAP analysis, feature importance, explainable AI  
**Software Engineering**: Modular code design, version control, reproducible pipelines  
**Python Development**: Pandas, NumPy, scikit-learn, CatBoost, SHAP, Matplotlib





