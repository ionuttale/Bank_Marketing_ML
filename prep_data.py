from ucimlrepo import fetch_ucirepo 
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

def missing_values(data):
    print("Missing values in each column:")
    print(data.isnull().sum())

def NaN_to_unknown(data):
    for column in data.columns:
        if data[column].isnull().any():
            data[column] = data[column].astype('category')
            data[column] = data[column].cat.add_categories(['unknown'])
            data[column] = data[column].fillna('unknown')
    return data

def remove_outliers(x : pd.DataFrame, y : pd.Series, columns : list) -> pd.DataFrame:
    
    mask = pd.Series(True, index=x.index)  # Start with all True

    for col in columns:
        Q1 = x[col].quantile(0.25)
        Q3 = x[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Update mask to keep only non-outliers
        mask &= x[col].between(lower, upper)

    # Apply mask to both x and y
    x_clean = x[mask].reset_index(drop=True)
    y_clean = y[mask].reset_index(drop=True)

    return x_clean, y_clean

def shuffle_data(x : pd.DataFrame, y : pd.Series) -> tuple:
    x, y = shuffle(x, y, random_state=42)
    return x, y

def undersample_data(x : pd.DataFrame, y : pd.Series) -> tuple:
    rus = RandomUnderSampler(random_state=42)
    x_resampled, y_resampled = rus.fit_resample(x, y)

    return x_resampled, y_resampled
