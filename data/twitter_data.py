import json
from pathlib import Path
import time

import pandas as pd
import tweepy

from data.secrets import twitter_config as secrets


RAW_DATA_DIR = Path('data/raw/Twitter-Waseem-2016')
BULLYING_LABELS = {'racism', 'sexism', 'both'}


def authenticate_with_twitter():
    # Authenticate
    auth = tweepy.OAuthHandler(secrets.API_KEY, secrets.API_SECRET_KEY)
    auth.set_access_token(secrets.ACCESS_TOKEN, secrets.ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    return api


def load_original_datasets():
    """Returns a preprocessed DataFrame from the original datasets"""
    # Preprocess 1st Dataset
    NAACL_SRW = pd.read_csv(RAW_DATA_DIR / 'NAACL_SRW_2016.csv',
                            index_col=0, header=None,
                            names=['TweetID', 'RawLabel'])
    NAACL_SRW['Label'] = NAACL_SRW['RawLabel'].isin(BULLYING_LABELS)

    # Preprocess 2nd Dataset
    NLP_CSS = pd.read_csv(RAW_DATA_DIR / 'NLP+CSS_2016.csv',
                          sep='\t', index_col=0)
    NLP_CSS = NLP_CSS[
        NLP_CSS.index.isin(set(NLP_CSS.index) - set(NAACL_SRW.index))
    ]
    NLP_CSS = NLP_CSS.dropna(axis=1, how='any')
    column_names = ['Expert', 'Amateur_0', 'Amateur_1', 'Amateur_2']
    NLP_CSS.columns = column_names
    NLP_CSS.index = NLP_CSS.index.rename('TweetID')
    # Create a label from majority voting of the labellers
    NLP_CSS['Label'] = (
        NLP_CSS.Expert.isin(BULLYING_LABELS) * 2 +
        sum(NLP_CSS[name].isin(BULLYING_LABELS) for name in column_names[1:])
    ) > 2

    # Concatenate the cleaned dataframes
    df = NLP_CSS.append(NAACL_SRW, sort=False)
    df['Text'] = pd.Series(dtype='object')
    return df


def get_tweets(api, df, ids, make_backup=False, file_name='twitter.csv'):
    """
    Uses the GET statuses/lookup to get the Tweet objects for a list of ids
    """
    i = 0
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
        texts = pd.Series(
            {tweet.id: getattr(tweet, 'text', None) for tweet in tweets}
        )
        df['Text'].update(texts)

        if make_backup:
            # Save json for backup
            for tweet in tweets:
                if len(tweet._json) > 1:
                    OUTPUT_DIR = RAW_DATA_DIR / 'tweets'
                    with (OUTPUT_DIR / f'{tweet.id_str}.json').open('w') as f:
                        f.write(json.dumps(tweet._json))

        # Increment counter
        i += 100

    if file_name:
        # Save to CSV if a file_name is supplied
        df[['Text', 'Label']].to_csv(Path('data') / file_name)

    return df


def load_dataset():
    df = pd.read_csv(Path('data') / 'twitter.csv', index_col=0)
    df = df.dropna()
    df.Label = df.Label.astype(bool)
    return df

# Tokenize
# Usernames: <USER_TOKEN>
# URLs: <URL_TOKEN>


if __name__ == "__main__":
    api = authenticate_with_twitter()
    df = load_original_datasets()
    list_of_ids = list(df.index)
    df = get_tweets(api, df, list_of_ids)
