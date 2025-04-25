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

def plot_all_boxplots(data: pd.DataFrame) -> None:
    # Create a figure with subplots (2 rows, 4 columns for 8 plots)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Boxplots of Features", fontsize=16)

    # List of columns to plot
    columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    titles = ['Age', 'Balance', 'Day of Week', 'Duration', 'Campaign', 'Pdays', 'Previous']

    # Iterate through columns and plot each boxplot
    for i, (col, title) in enumerate(zip(columns, titles)):
        row, col_idx = divmod(i, 4)  # Determine row and column index
        sns.boxplot(x=data[col], ax=axes[row, col_idx])
        axes[row, col_idx].set_title(title)

    # Hide any unused subplots (if fewer than 8 columns)
    for j in range(len(columns), 8):
        row, col_idx = divmod(j, 4)
        axes[row, col_idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.show()