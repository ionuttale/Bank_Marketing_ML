from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score, mean_squared_error
import data_preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

PCA_x_train = data_preprocessing.PCA_x_train
y_train = data_preprocessing.y_train
PCA_x_test = data_preprocessing.PCA_x_test
y_test = data_preprocessing.y_test

mlp = MLPClassifier(hidden_layer_sizes=(50,200,50), max_iter=300, activation='relu', solver='adam', random_state=1)
mlp.fit(PCA_x_train, y_train)

print(PCA_x_train)

print('Accuracy')
print(mlp.score(PCA_x_test, y_test))

predict = mlp.predict(PCA_x_test)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, predict)
fig, ax = plt.subplots(1)
ax = sns.heatmap(confusion_matrix, ax=ax, cmap=plt.cm.Blues, annot=True)
plt.ylabel('True value')
plt.xlabel('Predicted value')
plt.show()

print("Training error: %f" % mlp.loss_curve_[-1])
print("Training set score: %f" % mlp.score(PCA_x_train, y_train))
print("Test set score: %f" % mlp.score(PCA_x_test, y_test))
print(accuracy_score(y_test, predict))

print("MSE: %f" % mean_squared_error(y_test, predict))