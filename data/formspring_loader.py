from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path('data/raw/')
BULLYING_LABELS = {'Yes'}


def load_original_dataset():
    """Returns a preprocessed DataFrame from the original dataset"""
    # Preprocess Dataset
    data = pd.read_csv(RAW_DATA_DIR / 'formspring_data.csv', sep='\t')
    data['label'] = (
        sum(data[f'ans{i}'].isin(BULLYING_LABELS) for i in (1, 2, 3)) > 1
    )
    # Combine question and answer
    data['text'] = data.ques + '\n\n' + data.ans
    # Change index
    data.index = 'fs' + data.index.rename('id').astype(str)
    data = data[['text', 'label']].dropna()
    return data


def load_dataset():
    data = pd.read_csv(Path('data') / 'formspring.csv', index_col=0)
    return data
