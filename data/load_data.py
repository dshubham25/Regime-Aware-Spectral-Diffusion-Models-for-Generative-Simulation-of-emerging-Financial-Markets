import pandas as pd

def load_nifty(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df = df.sort_values('Date')

    df = df[['Date', 'Close', 'Shares Traded']]
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    return df