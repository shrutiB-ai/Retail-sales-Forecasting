import pandas as pd

def load_data(path):
    df=pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.dropna() # dropping null values for  now

    #feature engg
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Week'] = pd.to_datetime(df['Date']).dt.isocalendar().week

    #drop unused columns
    if 'Date' in df.columns:
        df = df.drop(columns = ['Date'])

    #catogorical column encoding
    if df['IsHoliday'].dtype =='bool':
        df['IsHoliday'] = df['IsHoliday'].astype(int)
    return df
