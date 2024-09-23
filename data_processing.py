# data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(uploaded_file):
    """Load the dataset from an uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    return df

def preprocess_data(df):
    """Convert 'Date' column to numerical format."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].map(pd.Timestamp.toordinal)
    return df

def split_data(df):
    """Split the data into features and target, and then into train and test sets."""
    X = df[['Date']].values
    y = df['Price'].values
    return train_test_split(X, y, test_size=0.2, random_state=42)