import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd

def check_age_outlier(data : pd.DataFrame) -> None:
    warnings.filterwarnings("ignore")
    fig, axes = plt.subplots(figsize=(9, 6))

    print("Age Shape:",data.shape)

    max_val = data.age.quantile(0.75)
    min_val = data.age.quantile(0.25)

    sns.boxplot(x = data['age'])
    plt.show()

def check_balance_outlier(data : pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(12, 7))

    # Checking the box plot for balance feature
    print("Balance Shape:",data.shape)
    ## Max and Min Quantile
    max_val = data.balance.quantile(0.75)
    min_val = data.balance.quantile(0.25)

    sns.boxplot(x = data['balance'])
    plt.show()
    print(min_val)

def check_day_outlier(data : pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(9, 6))

    # Checking the box plot for day feature
    print("Day Shape:",data.shape)
    ## Max and Min Quantile
    max_val = data.day_of_week.quantile(0.75)
    min_val = data.day_of_week.quantile(0.25)

    sns.boxplot(x = data['day_of_week'])
    plt.show()

def check_duration_outlier(data : pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(9, 6))

    # Checking the box plot for duration feature
    print("Duration Shape:",data.shape)
    ## Max and Min Quantile
    max_val = data.duration.quantile(0.75)
    min_val = data.duration.quantile(0.25)

    sns.boxplot(x = data['duration'])
    plt.show()

def check_campaign_outlier(data : pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(9, 6))

    # Checking the box plot for campaign feature
    print("Campaign Shape:",data.shape)
    ## Max and Min Quantile
    max_val = data.campaign.quantile(0.75)
    min_val = data.campaign.quantile(0.25)

    sns.boxplot(x = data['campaign'])
    plt.show()

def check_pdays_outlier(data : pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(9, 6))

    # Checking the box plot for pdays feature
    print("Pdays Shape:",data.shape)
    ## Max and Min Quantile
    max_val = data.pdays.quantile(0.75)
    min_val = data.pdays.quantile(0.25)

    sns.boxplot(x = data['pdays'])
    plt.show()

def check_previous_outlier(data : pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(9, 6))

    # Checking the box plot for previous feature
    print("Previous Shape:",data.shape)
    ## Max and Min Quantile
    max_val = data.previous.quantile(0.75)
    min_val = data.previous.quantile(0.25)

    sns.boxplot(x = data['previous'])
    plt.show()