import config
import data
import preprocessing
from helper import (
    create_learn_rate_scheduler, tensorboard_callback, early_stopping_callback
)
from helper import f1_score
from models import simple, bert_model


# Define callbacks

callbacks = [
        create_learn_rate_scheduler(max_learn_rate=1e-5,
                                    end_learn_rate=1e-7,
                                    warmup_epoch_count=10,
                                    total_epoch_count=config.EPOCH_COUNT),
        early_stopping_callback,
        tensorboard_callback
    ]


def main(model):
    df = preprocessing.preprocess(data.load("twitter"))
    df_train, df_validation = preprocessing.train_val_split(df)
    tokenizer = preprocessing.get_tokenizer(df_train)
    x_train, x_validation = preprocessing.tokenize(tokenizer,
                                                   df_train,
                                                   df_validation)

    y_train = df_train.label.to_numpy()
    y_validation = df_validation.label.to_numpy()

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', f1_score],
    )

    model.summary()
    history = model.fit(x=x_train, y=y_train,
                        batch_size=config.BATCH_SIZE,
                        shuffle=config.SHUFFLE,
                        epochs=config.EPOCH_COUNT,
                        validation_data=(x_validation, y_validation),
                        callbacks=callbacks)

    return history


if __name__ == "__main__":
    model = bert_model.model
    history = main(model)
    model.save_weights('./bert_cyberbullying_test.h5', overwrite=True)

    # To load weights for a compiled model
    # model.load_weights("bert_cyberbullying.h5")
