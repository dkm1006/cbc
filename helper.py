import math
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler


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
