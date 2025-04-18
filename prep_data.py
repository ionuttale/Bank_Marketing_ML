from random import randint
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
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

def shuffle_data(x : pd.DataFrame, y : pd.Series) -> tuple:
    x, y = shuffle(x, y, random_state=randint(0, 1000))

    return x, y

def undersample_data(x : pd.DataFrame, y : pd.Series) -> tuple:
    rus = RandomUnderSampler(random_state=42)
    x_resampled, y_resampled = rus.fit_resample(x, y)

    return x_resampled, y_resampled
