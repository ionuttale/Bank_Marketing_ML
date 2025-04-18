import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def age(data : pd.DataFrame) -> None:
    plt.figure(1, figsize=(9, 6))

    print("Age Shape:",data.shape)

    sns.boxplot(x = data['age'])
    plt.show()


def balance(data : pd.DataFrame) -> None:
    plt.figure(2, figsize=(9, 6))

    print("Balance Shape:",data.shape)

    sns.boxplot(x = data['balance'])
    plt.show()

def day(data : pd.DataFrame) -> None:
    plt.figure(3, figsize=(9, 6))

    print("Day Shape:",data.shape)

    sns.boxplot(x = data['day_of_week'])
    plt.show()

def duration(data : pd.DataFrame) -> None:
    plt.figure(4, figsize=(9, 6))

    print("Duration Shape:",data.shape)

    sns.boxplot(x = data['duration'])
    plt.show()

def campaign(data : pd.DataFrame) -> None:
    plt.figure(5, figsize=(9, 6))

    print("Campaign Shape:",data.shape)

    sns.boxplot(x = data['campaign'])
    plt.show()

def pdays(data : pd.DataFrame) -> None:
    plt.figure(6, figsize=(9, 6))

    print("Pdays Shape:",data.shape)

    sns.boxplot(x = data['pdays'])
    plt.show()

def previous(data : pd.DataFrame) -> None:
    plt.figure(7, figsize=(9, 6))

    print("Previous Shape:",data.shape)

    sns.boxplot(x = data['previous'])
    plt.show()