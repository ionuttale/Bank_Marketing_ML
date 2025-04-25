import pandas as pd
from sklearn.preprocessing import LabelEncoder
pd.set_option('future.no_silent_downcasting', True)

def education(data: pd.DataFrame) -> pd.DataFrame:
    education_mapping = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    data['education'] = data['education'].replace(education_mapping)
    return data

def duration(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(columns=['duration'], inplace=True, axis=1)
    return data

def categorical(data: pd.DataFrame) -> pd.DataFrame:
    categorial_features = ['job', 'marital', 'contact', 'month', 'poutcome']
    
    data = pd.get_dummies(data, columns=categorial_features, drop_first=True)
    
    print(data)

    return data

def binary(data: pd.DataFrame) -> pd.DataFrame:
    binary_mapping = {"yes": 1, "no": 0}
    data['y'] = data['y'].replace(binary_mapping).astype(int)
    data['default'] = data['default'].replace(binary_mapping).astype(int)
    data['housing'] = data['housing'].replace(binary_mapping).astype(int)
    data['loan'] = data['loan'].replace(binary_mapping).astype(int)
    return data

def encode(data: pd.DataFrame) -> pd.DataFrame:
    data = education(data)
    data = duration(data)
    data = categorical(data)
    data = binary(data)
    cols = list(data.columns.values)
    cols.pop(cols.index('y')) 
    data = data[cols+['y']]
    return data
