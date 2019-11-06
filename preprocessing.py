from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config


def preprocess(df):
    """Preprocess the DataFrame, replacing identifiable information"""
    # Usernames: <USER_TOKEN>
    username_pattern = r"(?<=\B|^)@\w{1,18}"
    df.text = df.text.str.replace(username_pattern, "<USERNAME>")
    # URLs: <URL_TOKEN>
    url_pattern = (
        r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]"
        r"|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    df.text = df.text.str.replace(url_pattern, "<URL>")
    # Email: <EMAIL_TOKEN>
    email_pattern = r"[-.+\w]+@[-\w]+\.[-.\w]+"
    df.text = df.text.str.replace(email_pattern, "<EMAIL>")
    # Replace tokens in Wikipedia Talk dataset
    df.text = df.text.str.replace("NEWLINE;?_TOKEN", "\n")
    df.text = df.text.str.replace("TAB_TOKEN", "\t")
    return df


def train_val_split(df):
    """Split the DataFrame in training and validation set"""
    num_training_samples = int(len(df) * config.TRAINING_PORTION)
    train_df = df[:num_training_samples]
    validation_df = df[num_training_samples:]
    return train_df, validation_df


def get_tokenizer(train_df):
    """Construct tokenizer and fit on training set"""
    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE,
                          oov_token=config.OOV_TOKEN)
    tokenizer.fit_on_texts(train_df.text)
    return tokenizer


def tokenize(tokenizer, train_df, validation_df):
    """Convert text to sequences and pad them"""
    train_sequences = tokenizer.texts_to_sequences(train_df.text)
    train_padded = pad_sequences(train_sequences,
                                 padding=config.PADDING_TYPE,
                                 truncating=config.TRUNC_TYPE,
                                 maxlen=config.MAX_LENGTH,)

    validation_sequences = tokenizer.texts_to_sequences(validation_df.text)
    validation_padded = pad_sequences(validation_sequences,
                                      padding=config.PADDING_TYPE,
                                      truncating=config.TRUNC_TYPE,
                                      maxlen=config.MAX_LENGTH)

    return train_padded, validation_padded


STOPWORDS = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "nor",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "would",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
]


# BUFFER_SIZE = 10000
# BATCH_SIZE = 64

# train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# train_dataset = train_dataset.padded_batch(BATCH_SIZE,
#                                            train_dataset.output_shapes)
