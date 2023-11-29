# Real-estate-price-predictor
# Machine Learning Project Documentation

## Overview

This project involves building a machine learning model to predict real estate prices. The process includes data analysis, handling missing values, selecting an appropriate model, training, evaluation, and saving the model.

## Project Setup

### Tools and Libraries

The project utilizes the following tools and libraries:
- Jupyter Notebook
- Pandas
- Numpy
- Scipy
- Scikit-learn
- Matplotlib

### Initial Setup

1. Jupyter Notebook:
   - Open Jupyter Notebook using the terminal command: `jupyter notebook`.

2. Library Installation:
   - Install necessary libraries: `jupyter, pandas, numpy, scipy, scikit-learn, matplotlib`.

## Data Analysis

1. Data Loading:
   - Downloaded the dataset as text, formatted it in Excel.
   - Read the dataset into a Pandas DataFrame:
     ```python
     import pandas as pd
     df = pd.read_csv("data.csv")
     ```

2. Exploratory Data Analysis:
   - Check top 5 rows: `df.head()`.
   - Basic info about the dataset: `df.info()`.
   - Count values of a specific feature: `housing_data['CHAS'].value_counts()`.
   - Descriptive statistics: `housing_data.describe()`.

3. Data Visualization:
   - Visualize data using Matplotlib:
     ```python
     import matplotlib.pyplot as plt
     %matplotlib inline
     housing_data.hist(bins=50, figsize=(20, 15))
     ```

4. Data Splitting:
   - Write a function for test-train split.
   - Utilize Scikit-learn for splitting:
     ```python
     from sklearn.model_selection import train_test_split
     train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)
     ```

   - Use StratifiedShuffleSplit for certain features:
     ```python
     from sklearn.model_selection import StratifiedShuffleSplit
     split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
     ```

## Correlation Analysis

1. Correlation Matrix:
   - Compute correlation matrix: `corr_matrix = housing_data.corr()`.
   - Sort correlations for a specific feature: `corr_matrix['MEDV'].sort_values(ascending=False)`.

2. Correlation Plots:
   - Plot scatter matrix: `scatter_matrix(housing_data[attributes], figsize=(12, 8))`.
   - Plot individual scatter plot: `housing_data.plot(kind='scatter', x='RM', y='MEDV', alpha=0.8)`.

3. Feature Engineering:
   - Create a new feature: `housing_data['TAXRM'] = housing_data['TAX'] / housing_data['RM']`.

## Data Preprocessing

1. Handling Missing Values:
   - Use SimpleImputer from Scikit-learn:
     ```python
     from sklearn.impute import SimpleImputer
     imputer = SimpleImputer(strategy='median')
     ```

   - Fill missing values: `new_housing_data = imputer.transform(housing_data)`.

2. Data Transformation:
   - Set up Scikit-learn pipeline for consistent data processing:
     ```python
     from sklearn.pipeline import Pipeline
     from sklearn.preprocessing import StandardScaler
     my_pipeline = Pipeline([
         ('imputer', SimpleImputer(strategy='median')),
         ('std_scaler', StandardScaler()),
     ])
     ```

## Model Selection and Training

1. Model Options:
   - Linear Regression
     ```python
     from sklearn.linear_model import LinearRegression
     model = LinearRegression()
     ```
   - Decision Tree
     ```python
     from sklearn.tree import DecisionTreeRegressor
     model = DecisionTreeRegressor()
     ```
   - Random Forest
     ```python
     from sklearn.ensemble import RandomForestRegressor
     model = RandomForestRegressor()
     ```

2. Training the Model:
   - Fit the model using pipeline:
     ```python
     model.fit(new_housing_data, housing_data_labels)
     ```

## Model Evaluation

1. Evaluation Metrics:
   - Mean Squared Error (MSE): `mean_squared_error(housing_data_labels, housing_predictions)`.

2. Cross-Validation:
   - Apply cross-validation to avoid overfitting:
     ```python
     from sklearn.model_selection import cross_val_score
     scores = cross_val_score(model, new_housing_data, housing_data_labels, scoring="neg_mean_squared_error", cv=10)
     ```

   - Display scores and statistics:
     ```python
     print_scores(rmse_scores)
     ```

## Model Saving and Testing

1. Save Model:
   - Use joblib to save the model:
     ```python
     from joblib import dump
     dump(model, 'Realestate_valuepredictor.joblib')
     ```

2. Test Model on New Data:
   - Load the saved model:
     ```python
     from joblib import load
     model = load('Realestate_valuepredictor.joblib')
     ```

   - Make predictions on new data:
     ```python
     feature_values = np.array([[-0.43942006, 3.12628155, ...]])
     model.predict(feature_values)
     ```

## Conclusion

This documentation outlines the entire process of building a real estate price prediction model, from data analysis to model evaluation. The chosen model can be saved and used for predicting prices on new data. For more details, refer to the Jupyter Notebook.
