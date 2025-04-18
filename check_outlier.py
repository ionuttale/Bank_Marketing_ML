import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def age(data : pd.DataFrame) -> None:
    plt.figure(1, figsize=(9, 6))

    print("Age Shape:",data.shape)

    max_val = data.age.quantile(0.75)
    min_val = data.age.quantile(0.25)

    sns.boxplot(x = data['age'])
    plt.show()

    print("Max Age:",max_val)
    print("Min Age:",min_val)

def balance(data : pd.DataFrame) -> None:
    plt.figure(2, figsize=(12, 7))

    print("Balance Shape:",data.shape)

    max_val = data.balance.quantile(0.75)
    min_val = data.balance.quantile(0.25)

    sns.boxplot(x = data['balance'])
    plt.show()

    print("Max Balance:",max_val)
    print("Min Balance:",min_val)

def day(data : pd.DataFrame) -> None:
    plt.figure(3, figsize=(9, 6))

    print("Day Shape:",data.shape)

    max_val = data.day_of_week.quantile(0.75)
    min_val = data.day_of_week.quantile(0.25)

    sns.boxplot(x = data['day_of_week'])
    plt.show()

    print("Max Day:",max_val)
    print("Min Day:",min_val)

def duration(data : pd.DataFrame) -> None:
    plt.figure(4, figsize=(9, 6))

    print("Duration Shape:",data.shape)

    max_val = data.duration.quantile(0.75)
    min_val = data.duration.quantile(0.25)

    sns.boxplot(x = data['duration'])
    plt.show()

    print("Max Duration:",max_val)
    print("Min Duration:",min_val)

def campaign(data : pd.DataFrame) -> None:
    plt.figure(5, figsize=(9, 6))

    print("Campaign Shape:",data.shape)

    max_val = data.campaign.quantile(0.75)
    min_val = data.campaign.quantile(0.25)

    sns.boxplot(x = data['campaign'])
    plt.show()

    print("Max Campaign:",max_val)
    print("Min Campaign:",min_val)

def pdays(data : pd.DataFrame) -> None:
    plt.figure(6, figsize=(9, 6))

    print("Pdays Shape:",data.shape)

    max_val = data.pdays.quantile(0.75)
    min_val = data.pdays.quantile(0.25)

    sns.boxplot(x = data['pdays'])
    plt.show()

    print("Max Pdays:",max_val)
    print("Min Pdays:",min_val)

def previous(data : pd.DataFrame) -> None:
    plt.figure(7, figsize=(9, 6))

    print("Previous Shape:",data.shape)

    max_val = data.previous.quantile(0.75)
    min_val = data.previous.quantile(0.25)

    sns.boxplot(x = data['previous'])
    plt.show()

    print("Max Previous:",max_val)
    print("Min Previous:",min_val)