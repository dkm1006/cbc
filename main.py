import data
import preprocessing
from models import simple, bert_model

df = preprocessing.preprocess(data.load("twitter"))
train_df, validation_df = preprocessing.train_val_split(df)
tokenizer = preprocessing.get_tokenizer(train_df)
train_padded, validation_padded = preprocessing.tokenize(tokenizer,
                                                         train_df,
                                                         validation_df)

history = simple.model.fit(x=train_padded,
                           y=train_df.label.to_numpy(),
                           epochs=2)
eval_loss, eval_acc = simple.model.evaluate(x=validation_padded,
                                            y=validation_df.label.to_numpy())
