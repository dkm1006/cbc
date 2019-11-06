import tensorflow as tf
from tensorflow.keras import layers
import config
import data
import preprocessing

model = tf.keras.Sequential(
    [
        layers.Embedding(input_dim=config.VOCAB_SIZE,
                         output_dim=config.EMBEDDING_DIM,
                         input_length=config.MAX_LENGTH),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)


if __name__ == "__main__":
    df = preprocessing.preprocess(data.load("twitter"))
    train_df, validation_df = preprocessing.train_val_split(df)
    tokenizer = preprocessing.get_tokenizer(train_df)
    train_padded, validation_padded = preprocessing.tokenize(tokenizer,
                                                             train_df,
                                                             validation_df)
    history = model.fit(x=train_padded, y=train_df.label.to_numpy(), epochs=2)
    eval_loss, eval_acc = model.evaluate(x=validation_padded,
                                         y=validation_df.label.to_numpy())
