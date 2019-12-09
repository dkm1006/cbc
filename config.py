VOCAB_SIZE = 10000  # Could use a larger vocab
OOV_TOKEN = "<OOV>"  # Maybe use [UNK]
MAX_LENGTH = 128  # 160 is better for tweets
PADDING_TYPE = "post"
TRUNC_TYPE = "post"
TRAINING_PORTION = 0.8
EMBEDDING_DIM = 64

SHUFFLE = True
BATCH_SIZE = 16
EPOCH_COUNT = 20
