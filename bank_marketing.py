from ucimlrepo import fetch_ucirepo 
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
import prep_data
import check_outliers


bank_marketing = fetch_ucirepo(id=222)

x = bank_marketing.data.features  
y = bank_marketing.data.targets   

x = prep_data.NaN_to_unknown(x)
y = prep_data.NaN_to_unknown(y)

x, y = prep_data.shuffle_data(x, y)

check_outliers.check_age_outlier(x)
check_outliers.check_balance_outlier(x)
check_outliers.check_day_outlier(x)
check_outliers.check_duration_outlier(x)
check_outliers.check_pdays_outlier(x)
check_outliers.check_campaign_outlier(x)
check_outliers.check_previous_outlier(x)

