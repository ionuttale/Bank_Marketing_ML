from ucimlrepo import fetch_ucirepo 
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
import prep_data

def print_data(data):
    print("Data:")
    print(data)

def __init__():
    bank_marketing = fetch_ucirepo(id=222)
    
    x = bank_marketing.data.features  
    y = bank_marketing.data.targets   

    x = prep_data.NaN_to_unknown(x)
    y = prep_data.NaN_to_unknown(y)

    x, y = prep_data.shuffle_data(x, y)

    x, y = prep_data.remove_outliers(x, y, ['age', 'balance', 'campaign'])

    # x, y = prep_data.undersample_data(x, y)

    # Trebuie sa facem encoding la datele de tip object

    print_data(x)
    print_data(y)

__init__()
