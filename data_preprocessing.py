from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import remove_outlier
import seaborn as sns
import matplotlib.pyplot as plt
import encode_data

data = pd.read_csv('bank-full.csv', sep=';')

data = remove_outlier.balance(data)
data = remove_outlier.duration(data)
data = remove_outlier.pdays(data)
data = remove_outlier.campaign(data)
data = remove_outlier.previous(data)

data = encode_data.encode(data)

y = data['y']
x = data.values[:, :-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

correlation_matrix = pd.DataFrame(x_train).corr()
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(correlation_matrix, ax=ax)
plt.show()

# getting the upper triangle of the correlation matrix
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool))
print(upper_tri)

# checking which columns can be dropped
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print('\nTo drop')
print(to_drop)

# removing the selected columns
x_train = x_train.drop(x_train.columns[to_drop], axis=1)
x_test = x_test.drop(x_test.columns[to_drop], axis=1)
print(x_train.head())

# apply the PCA for feature for feature reduction
pca = PCA(n_components=0.95)
pca.fit(x_train)
PCA_x_train = pca.transform(x_train)
PCA_x_test = pca.transform(x_test)

print(x_train)