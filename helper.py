import math
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import (
    LearningRateScheduler, TensorBoard, EarlyStopping
)
import tensorflow.keras.backend as kb

LOG_DIR = ".log/" + datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = TensorBoard(log_dir=LOG_DIR)
early_stopping_callback = EarlyStopping(patience=20, restore_best_weights=True)


def create_learn_rate_scheduler(max_learn_rate=5e-5,
                                end_learn_rate=1e-7,
                                warmup_epoch_count=10,
                                total_epoch_count=90):

    def calculate_learn_rate(epoch):
        if epoch < warmup_epoch_count:
            lr = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
        else:
            log_ratio = math.log(end_learn_rate / max_learn_rate)
            numerator = log_ratio * (epoch - warmup_epoch_count + 1)
            denominator = (total_epoch_count - warmup_epoch_count + 1)
            exponent = numerator / denominator
            lr = max_learn_rate * math.exp(exponent)
        return float(lr)

    learn_rate_scheduler = LearningRateScheduler(calculate_learn_rate)
    return learn_rate_scheduler


def adjusted_sum(x, a=0, b=1):
    return kb.sum(kb.round(kb.clip(x, a, b)))


def f1_score(y_true, y_pred):
    true_positives = adjusted_sum(y_true * y_pred)
    real_positives = adjusted_sum(y_true)
    predicted_positives = adjusted_sum(y_pred)
    precision = true_positives / (predicted_positives + kb.epsilon())
    recall = true_positives / (real_positives + kb.epsilon())
    return 2 * (precision * recall) / (precision + recall + kb.epsilon())
