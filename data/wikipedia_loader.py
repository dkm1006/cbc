from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path('data') / 'raw' / 'wikipedia-talk-labels-personal-attacks'


def load_original_dataset():
    """Returns a preprocessed DataFrame from the original datasets"""
    # Read original files
    comments_file = RAW_DATA_DIR / 'attack_annotated_comments.csv'
    labels_file = RAW_DATA_DIR / 'attack_annotations.csv'
    comments = pd.read_csv(comments_file, index_col=0)
    labels = pd.read_csv(labels_file, index_col=[0, 1])
    # The 'attack' column is 1 whenever there is one kind of attack
    assert (labels[labels.columns[:-1]].any(axis=1) == labels['attack']).all()
    comments['label'] = (labels.mean(level=0).attack >= 0.5)
    comments.index = 'wi' + comments.index.rename('id').astype(str)
    df = comments[['comment', 'label']]
    df.columns = ['text', 'label']
    return df

# TODO: Deal with NEWLINE_TOKEN


def load_dataset():
    df = pd.read_csv(Path('data') / 'wikipedia.csv', index_col=0)
    return df
