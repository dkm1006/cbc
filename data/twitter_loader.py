import json
from pathlib import Path
import time

import pandas as pd
import tweepy

import secrets

DATA_DIR = Path('data')
RAW_DATA_DIR = DATA_DIR / 'raw'
BULLYING_LABELS = {'racism', 'sexism', 'both'}


def authenticate_with_twitter():
    # Authenticate
    auth = tweepy.OAuthHandler(secrets.TWITTER_API_KEY,
                               secrets.TWITTER_API_SECRET_KEY)
    auth.set_access_token(secrets.TWITTER_ACCESS_TOKEN,
                          secrets.TWITTER_ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    return api


def load_original_datasets():
    """Returns a preprocessed DataFrame from the original datasets"""
    WASEEM_DIR = RAW_DATA_DIR / 'Twitter-Waseem-2016'
    # Preprocess 1st Dataset
    NAACL_SRW = pd.read_csv(WASEEM_DIR / 'NAACL_SRW_2016.csv',
                            index_col=0, header=None,
                            names=['id', 'RawLabel'])
    NAACL_SRW['label'] = NAACL_SRW['RawLabel'].isin(BULLYING_LABELS)

    # Preprocess 2nd Dataset
    NLP_CSS = pd.read_csv(WASEEM_DIR / 'NLP+CSS_2016.csv',
                          sep='\t', index_col=0)
    NLP_CSS = NLP_CSS[
        NLP_CSS.index.isin(set(NLP_CSS.index) - set(NAACL_SRW.index))
    ]
    NLP_CSS = NLP_CSS.dropna(axis=1, how='any')
    column_names = ['Expert', 'Amateur_0', 'Amateur_1', 'Amateur_2']
    NLP_CSS.columns = column_names
    NLP_CSS.index = NLP_CSS.index.rename('id')
    # Create a label from majority voting of the labellers
    NLP_CSS['label'] = (
        NLP_CSS.Expert.isin(BULLYING_LABELS) * 2 +
        sum(NLP_CSS[name].isin(BULLYING_LABELS) for name in column_names[1:])
    ) > 2

    # Concatenate the cleaned dataframes
    df = NLP_CSS.append(NAACL_SRW, sort=False)
    return df


def get_tweets(api, ids, make_backup=False):
    """
    Uses the GET statuses/lookup to get the Tweet objects for a list of ids
    """
    i = 0
    texts = pd.Series({str(id): '' for id in ids}, dtype='object')
    while i < len(ids):
        try:
            tweets = api.statuses_lookup(ids[i:i+100],
                                         include_entities=True,
                                         trim_user=True, map_=True)
        except tweepy.RateLimitError:
            # Sleep for 15 minutes and then continue
            print('Hit rate limit. Sleeping for 15 minutes.')
            time.sleep(15 * 60)
            continue

        # Extract texts and update DataFrame
        new_texts = pd.Series(
            {str(tweet.id): getattr(tweet, 'text', None) for tweet in tweets}
        )
        texts.update(new_texts)

        if make_backup:
            # Save json for backup
            for tweet in tweets:
                if len(tweet._json) > 1:
                    OUTPUT_DIR = RAW_DATA_DIR / 'tweets'
                    with (OUTPUT_DIR / f'{tweet.id_str}.json').open('w') as f:
                        f.write(json.dumps(tweet._json))

        # Increment counter
        i += 100

    return texts


def load_dataset(file_name='twitter.csv'):
    df = pd.read_csv(DATA_DIR / file_name, index_col=0)
    df = df.dropna()
    df.label = df.label.astype(bool)
    return df


def load_sui_dataset():
    # No overlap with the other datasets
    SUI_DIR = RAW_DATA_DIR / 'twitterbullyingV3'
    columns = {
        'id': str,
        'user': str,
        'label': bool,
        'type': 'category',
        'form': 'category',
        'teasing': bool,
        'role': 'category',
        'emotion': 'category'
    }
    converters = {
        'label': lambda v: True if v == 'y' else False,
        'teasing': lambda v: True if v == 'y' else False
    }
    SUI = pd.read_csv(SUI_DIR / 'data.csv',
                      names=columns.keys(),
                      index_col=0,
                      dtype=columns,
                      converters=converters)
    SUI.index = SUI.index.astype(str)
    SUI = SUI[~SUI.index.duplicated()]

    # TODO: Take into account the role. Only take the attacker,
    # otherwise we get stuff like 'Bully' classified as bullying
    return SUI


if __name__ == "__main__":
    api = authenticate_with_twitter()
    df = load_original_datasets()
    list_of_ids = list(df.index)
    df['text'] = get_tweets(api, list_of_ids)
    # Save to CSV if a file_name is supplied
    df[['text', 'label']].to_csv(DATA_DIR / 'twitter-clean.csv')
