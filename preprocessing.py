from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bert.tokenization import FullTokenizer as BertTokenizer
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


def train_val_split(df, **kwargs):
    """Split the DataFrame in training and validation set"""
    num_split = int(len(df) * config.TRAINING_PORTION)
    shuffled = df.sample(frac=1, **kwargs)
    train_df, validation_df = shuffled[:num_split], shuffled[num_split:]
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
                                 maxlen=config.MAX_LENGTH)

    validation_sequences = tokenizer.texts_to_sequences(validation_df.text)
    validation_padded = pad_sequences(validation_sequences,
                                      padding=config.PADDING_TYPE,
                                      truncating=config.TRUNC_TYPE,
                                      maxlen=config.MAX_LENGTH)

    return train_padded, validation_padded


# From Albert Tokenization ####################################################


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            token = line.strip().split()[0]
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def convert_tokens_to_ids(vocab, tokens):
    return [vocab[token] for token in tokens]


def convert_ids_to_tokens(inv_vocab, ids):
    return [inv_vocab[id] for id in ids]


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    raw_tokens = text.split()
    # tokens = [token.strip() for token in raw_tokens]
    return raw_tokens


class FullTokenizer:
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_ids_to_tokens(self.inv_vocab, ids)


class BasicTokenizer:
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
        Args:
        do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
        output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                    start_new_word = False
                    output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _clean_text(self, text):
        """Removes invalid characters and cleans up whitespace in text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer:
    """Runs WordPiece tokenziation."""
    # TODO: Change unk_token to oov_token='<OOV>'
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to tokenize
        using the given vocabulary.
        For example:
        input = "unaffable"
        output = ["un", "##aff", "##able"]
        Args:
        text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
        A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + six.ensure_str(substr)
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((33 <= cp <= 47) or (58 <= cp <= 64) or
            (91 <= cp <= 96) or (123 <= cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def create_tokenizer_from_hub_module(albert_hub_module_handle):
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        albert_module = hub.Module(albert_hub_module_handle)
        tokenization_info = albert_module(signature="tokenization_info",
                                          as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run(
                [tokenization_info["vocab_file"],
                 tokenization_info["do_lower_case"]]
            )
    return tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case,
        spm_model_file=FLAGS.spm_model_file)


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
