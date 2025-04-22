from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import data_preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Data preparation
PCA_x_train = data_preprocessing.PCA_x_train
y_train = data_preprocessing.y_train
PCA_x_test = data_preprocessing.PCA_x_test
y_test = data_preprocessing.y_test

# Generate experiment configurations
hidden_layers_options = [1, 2]
neurons_options = [50, 100, 200]
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
                'hidden_layers': (neurons, neurons * 2),
                'learning_rate': learning_rates
            })

# Run experiments
for config in configurations:
    for lr in config['learning_rate']:
        print(f"Running configuration: Hidden Layers={config['hidden_layers']}, Learning Rate={lr}")
        
        # Initialize and train the model
        mlp = MLPClassifier(hidden_layer_sizes=config['hidden_layers'], max_iter=300, activation='relu', solver='adam', learning_rate_init=lr, random_state=1)
        mlp.fit(PCA_x_train, y_train)

        # Evaluate the model
        predict = mlp.predict(PCA_x_test)
        acc = accuracy_score(y_test, predict)
        mse = mean_squared_error(y_test, predict)
        print(f"Accuracy: {acc}, MSE: {mse}")

        # Confusion matrix
        cm = confusion_matrix(y_test, predict)
        fig, ax = plt.subplots(1)
        sns.heatmap(cm, ax=ax, cmap=plt.cm.Blues, annot=True)
        plt.title(f"Confusion Matrix (Hidden Layers={config['hidden_layers']}, LR={lr})")
        plt.ylabel('True value')
        plt.xlabel('Predicted value')
        plt.show()

        print("Training error: %f" % mlp.loss_curve_[-1])
        print("Training set score: %f" % mlp.score(PCA_x_train, y_train))
        print("Test set score: %f" % mlp.score(PCA_x_test, y_test))