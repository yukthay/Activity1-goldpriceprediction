import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta

def train_model(X_train, y_train):
    """Initialize and train the Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'R² Score': r2
    }
    return metrics

def forecast_prices(model, df, days=30):
    """Forecast prices for the next 'days' days."""
    last_date_ordinal = df['Date'].max()
    last_date = datetime.fromordinal(last_date_ordinal)
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]

    future_dates_ordinal = [date.toordinal() for date in future_dates]
    future_dates_array = np.array(future_dates_ordinal).reshape(-1, 1)

    future_prices = model.predict(future_dates_array)

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Price': future_prices
    })
    return forecast_df
