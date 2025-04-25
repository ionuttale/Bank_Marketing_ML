from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import data_preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

x_train = data_preprocessing.x_train
y_train = data_preprocessing.y_train
x_test = data_preprocessing.x_test
y_test = data_preprocessing.y_test

# print(x_train)
# print(x_test.shape)

hidden_layers_options = [1, 2]
neurons_options = [12]
learning_rates = [0.1, 0.01]

configurations = []
for hidden_layers in hidden_layers_options:
    for neurons in neurons_options:
        if hidden_layers == 1:
            configurations.append({
                'hidden_layers': (neurons,),
                'learning_rate': learning_rates
            })
        elif hidden_layers == 2:
            configurations.append({
                'hidden_layers': (neurons, neurons // 2),
                'learning_rate': learning_rates
            })
            configurations.append({
                'hidden_layers': (neurons, neurons),
                'learning_rate': learning_rates
            })
            configurations.append({
                'hidden_layers': (neurons, neurons * 2),
                'learning_rate': learning_rates
            })

for config in configurations:
    for lr in config['learning_rate']:
        print(f"Running configuration: Hidden Layers={config['hidden_layers']}, Learning Rate={lr}")
        
        mlp = MLPClassifier(hidden_layer_sizes=config['hidden_layers'], max_iter=2000, activation='relu', solver='adam', learning_rate_init=lr, random_state=1)
        mlp.fit(x_train, y_train)

        predict = mlp.predict(x_test)
        acc = accuracy_score(y_test, predict)
        mse = mean_squared_error(y_test, predict)
        print(f"Accuracy: {acc}, MSE: {mse}")

        cm = confusion_matrix(y_test, predict)
        fig, ax = plt.subplots(1)
        sns.heatmap(cm, ax=ax, cmap=plt.cm.Blues, annot=True)
        # plt.title(f"Confusion Matrix (Hidden Layers={config['hidden_layers']}, LR={lr})")
        # plt.ylabel('True value')
        # plt.xlabel('Predicted value')
        # plt.show()
