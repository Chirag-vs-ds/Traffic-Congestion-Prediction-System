import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df
