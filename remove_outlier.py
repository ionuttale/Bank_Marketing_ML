import pandas as pd

def balance(data : pd.DataFrame) -> None:
    data = data[(data['balance']>-6000) & (data['balance']<50000)]
    return data

def duration(data : pd.DataFrame) -> None:
    data = data[(data['duration']<2600)] 
    return data

def pdays(data : pd.DataFrame) -> None:
    data = data[(data['pdays']<575)]
    return data

def campaign(data : pd.DataFrame) -> None:
    data = data[(data['campaign']<40)]
    return data

def previous(data : pd.DataFrame) -> None:
    data = data[(data['previous']<50)]
    return data