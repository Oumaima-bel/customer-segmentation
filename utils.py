import pandas as pd


def load_df():
    df = pd.read_csv('dataset/marketing_campaign.csv', sep='\t')
    return df