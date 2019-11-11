from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')
DIR = DATA_DIR / 'raw' / 'jigsaw-unintended-bias-in-toxicity-classification'


def load_original_dataset():
    df = pd.read_csv(DIR / 'train.csv', index_col=0)
    df['label'] = (df.target >= 0.5)
    df = df[['comment_text', 'label']]
    df.columns = ['text', 'label']
    df.index = 'ji' + df.index.astype(str)
    return df


def load_dataset():
    df = pd.read_csv(DATA_DIR / 'jigsaw.csv', index_col=0)
    return df
