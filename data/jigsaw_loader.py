from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')
DIR = DATA_DIR / 'raw' / 'jigsaw-unintended-bias-in-toxicity-classification'


def load_original_dataset():
    data = pd.read_csv(DIR / 'train.csv', index_col=0)
    data['label'] = (data.target >= 0.5)
    data = data[['comment_text', 'label']]
    data.columns = ['text', 'label']
    data.index = 'ji' + data.index.astype(str)
    return data


def load_dataset():
    data = pd.read_csv(DATA_DIR / 'jigsaw.csv', index_col=0)
    return data
