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
    train_df, validation_df = preprocessing.train_val_split(df)
    tokenizer = preprocessing.get_tokenizer(train_df)
    train_padded, validation_padded = preprocessing.tokenize(tokenizer,
                                                             train_df,
                                                             validation_df)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', f1_score],
    )

    model.summary()
    history = model.fit(x=train_padded, y=train_df.label.to_numpy(),
                        # validation_split=0.1,
                        batch_size=config.BATCH_SIZE,
                        shuffle=config.SHUFFLE,
                        epochs=config.EPOCH_COUNT,
                        callbacks=callbacks)
    eval_loss, eval_acc = model.evaluate(x=validation_padded,
                                         y=validation_df.label.to_numpy())

    return history, eval_loss, eval_acc


if __name__ == "__main__":
    model = bert_model.model
    history, eval_loss, eval_acc = main(model)
    model.save_weights('./bert_cyberbullying_test.h5', overwrite=True)

    # To load weights for a compiled model
    # model.load_weights("bert_cyberbullying.h5")
