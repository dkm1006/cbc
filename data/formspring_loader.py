from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path('data/raw/')
BULLYING_LABELS = {'Yes'}


def load_original_dataset():
    """Returns a preprocessed DataFrame from the original dataset"""
    # Preprocess Dataset
    df = pd.read_csv(RAW_DATA_DIR / 'formspring_data.csv', sep='\t')
    df['label'] = (
        sum(df[f'ans{i}'].isin(BULLYING_LABELS) for i in (1, 2, 3)) > 1
    )
    # Combine question and answer
    df['text'] = df.ques + '\n\n' + df.ans
    # Change index
    df.index = 'fs' + df.index.rename('id').astype(str)
    df = df[['text', 'label']].dropna()
    return df


def load_dataset():
    df = pd.read_csv(Path('data') / 'formspring.csv', index_col=0)
    return df
