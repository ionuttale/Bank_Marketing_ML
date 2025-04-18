import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

def education(data: pd.DataFrame) -> pd.DataFrame:
    
    # Create a mapping dictionary for education levels
    education_mapping = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}

    data['education'] = data['education'].replace(education_mapping)

    return data

def duration(data: pd.DataFrame) -> pd.DataFrame:
    
    data.drop(columns=['duration'], inplace=True, axis=1)

    return data

def categorical(data: pd.DataFrame) -> pd.DataFrame:
    
    categorial_features = ['job', 'marital', 'contact', 'month', 'poutcome']
    # categorial_features = ['job', 'marital', 'poutcome']
    for item in categorial_features:
        # assigning the encoded data into a new DataFrame object
        df = pd.get_dummies(data[item], prefix=item)
        data = data.drop(item, axis=1)
        for categorial_feature in df.columns:
            #Set the new column in data to have corresponding df values
            data[categorial_feature] = df[categorial_feature]
    
    return data

def binary(data: pd.DataFrame) -> pd.DataFrame:
    
    # Create a mapping dictionary for binary variables
    binary_mapping = {"yes": 1, "no": 0}
    
    # Apply the mapping to the binary columns
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