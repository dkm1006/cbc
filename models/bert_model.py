import bert
from bert.tokenization import FullTokenizer as BertTokenizer
from tensorflow import keras


MAX_SEQ_LEN = 128
ADAPTER_SIZE = None  # Use None for Fine-Tuning
MODEL_NAME = "albert_base"
MODEL_URL = 'https://tfhub.dev/google/albert_base/2?tf-hub-format=compressed'
CHECKPOINT_DIR = 'checkpoints'
MODEL_DIR = bert.fetch_tfhub_albert_model(MODEL_NAME, CHECKPOINT_DIR)


def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        yield from flatten_layers(layer)


def freeze_layers(root_layer, exclude=None):
    exclude = [] if exclude is None else exclude
    root_layer.trainable = False
    for layer in flatten_layers(root_layer):
        if layer.name in exclude:
            layer.trainable = True


# Create ALBERT layer
model_params = bert.albert_params(MODEL_NAME)
model_params.adapter_size = ADAPTER_SIZE
bert_layer = bert.BertModelLayer.from_params(model_params, name="albert")


# Define model architecture
input_ids = keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32')
# NOTE: Following line not required if using default token_type/segment id 0
# token_type_ids = keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32')
output = bert_layer(input_ids)  # output:[batch_size, MAX_SEQ_LEN, hidden_size]
# NOTE: The following is an alternative for classification taken from
# https://github.com/kpe/bert-for-tf2/blob/master/examples/gpu_movie_reviews.ipynb
# The Lambda layer just takes one output from the sequence
cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
# TODO: Try with more regularisation
# cls_out = keras.layers.Dropout(rate=0.5)(cls_out)
logits = keras.layers.Dense(units=256, activation='relu')(cls_out)
logits = keras.layers.Dropout(rate=0.2)(logits)
# NOTE: Alternative to the Lambda layer
# bgru_layer = keras.layers.Bidirectional(keras.layers.GRU(64))(output)
output = keras.layers.Dense(units=1, activation='sigmoid')(logits)
model = keras.Model(inputs=input_ids, outputs=output)

# Freeze all non-trainable layers
freeze_layers(bert_layer, exclude=['LayerNorm'])
# Originally from tutorial: ['LayerNorm', 'adapter-down', 'adapter-up']

# Build model and load pre-trained weights
model.build(input_shape=(None, MAX_SEQ_LEN))
bert.load_albert_weights(bert_layer, MODEL_DIR)


# Alternative for loading weights from checkpoint file

# from bert.loader import (StockBertConfig, map_stock_config_to_params,
#                          load_stock_weights)
# bert_ckpt_dir="gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12/"
# bert_ckpt_file = bert_ckpt_dir + "bert_model.ckpt"
# bert_config_file = bert_ckpt_dir + "bert_config.json"
# bert_model_dir="2018_10_18"
# bert_model_name="uncased_L-12_H-768_A-12"

# Read bert_layer parameters from config_file
# with open(bert_config_file, 'r') as reader:
#     bert_config = StockBertConfig.from_json_string(reader.read())
#     bert_params = map_stock_config_to_params(bert_config)
#     bert_params.adapter_size = ADAPTER_SIZE
#     bert_layer = BertModelLayer.from_params(bert_params, name="bert")

# Load the pre-trained model weights
# load_stock_weights(bert_layer, bert_ckpt_file)


if __name__ == "__main__":
    import data
    import preprocessing
    file_path = 'checkpoints/wwm_uncased_L-24_H-1024_A-16/vocab.txt'
    tokenizer = BertTokenizer(vocab_file=file_path, do_lower_case=True)
    df = data.load("twitter")
    df = preprocessing.preprocess(df)

    # TODO: Truncate before adding CLS and SEP
    # TODO: Anonymisation of URLs and USERNAMEs etc. is currently shitty. 
    #       Try adding to vocab file.
    tokens = df.text.apply(tokenizer.tokenize)
    tokens = tokens.apply(lambda x: ['[CLS]'] + x + ['[SEP]'])
    ids = tokens.apply(tokenizer.convert_tokens_to_ids)
