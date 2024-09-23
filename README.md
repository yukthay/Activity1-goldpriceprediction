# Gold Price Prediction and Forecasting
This project is a Streamlit web application that predicts and forecasts gold prices using a Linear Regression model. The app allows users to upload a CSV file containing historical gold prices, trains a machine learning model, evaluates its performance, and provides a 30-day price forecast.


# Features
- Upload CSV: Upload a CSV file containing gold price data with columns: `Date` and `Price`.
- Data Preprocessing: Converts the date column to a numerical format suitable for model training.
- Model Training: Trains a Linear Regression model on the historical data.
- Model Evaluation: Provides evaluation metrics including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
- Forecasting: Generates a 30-day forecast of future gold prices.

# File-wise Explanation

# app.py
- This is the main file that runs the Streamlit application. It serves as the entry point of the app and connects the user interface with the data processing and machine learning functionalities.

Responsibilities:
    -Loads the CSV file uploaded by the user.
    -Calls functions for data preprocessing, model training, evaluation, and forecasting.
    -Displays forecasted Gold prices in the Streamlit interface.

# data_processing.py
- This file handles the core data-related operations like loading, preprocessing, and splitting the dataset.

Functions:
    -load_data(file_path): Loads the dataset from the uploaded CSV file.
    -preprocess_data(df): Preprocesses the data by converting the Date column to a numerical format.
    -split_data(df): Splits the data into features and target variables, and then into training and test sets.

# model.py
- This file contains all the machine learning functionality, including model training, evaluation, and forecasting.

Functions:
    -train_model(X_train, y_train): Trains a Linear Regression model on the provided training data.
    -evaluate_model(model, X_test, y_test): Evaluates the model on the test data and returns metrics like MAE, MSE, RMSE, and R².
    -forecast_prices(model, df, days=30): Predicts gold prices for the next 60 days based on the trained model.