import os

# Keys and tokens
# Keys, secret keys and access tokens management.

# Consumer API keys
TWITTER_API_KEY = os.environ.get('TWITTER_API_KEY', '')
TWITTER_API_SECRET_KEY = os.environ.get('TWITTER_API_SECRET_KEY', '')

# Access token & access token secret / Read and write (Access level)
TWITTER_ACCESS_TOKEN = os.environ.get('TWITTER_ACCESS_TOKEN', '')
TWITTER_ACCESS_TOKEN_SECRET = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET', '')
