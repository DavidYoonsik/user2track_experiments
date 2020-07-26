import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from src.generator.data_generator import DataGenerator
from src.model.create_train_model import user2track_model

np.random.seed(7)


def u2t_train(x_play_train, x_skip_train, y_play_train, x_play_test, x_skip_test, y_play_test, track_index, batch_size,
              num_inputs, num_outputs, neg_k, lr, embedding, freq_prob):
    params_train = {
        'neg_k'     : neg_k,
        'batch_size': batch_size,
        'num_inputs': num_inputs,
        'shuffle'   : True
        }
    params_val = {
        'neg_k'     : neg_k,
        'batch_size': batch_size,
        'num_inputs': num_inputs,
        'shuffle'   : False
        }
    
    train_generator = DataGenerator(x_play_train, x_skip_train, y_play_train, track_index, freq_prob, **params_train)
    validation_generator = DataGenerator(x_play_test, x_skip_test, y_play_test, track_index, freq_prob, **params_val)
    
    sess = tf.Session()
    K.set_session(session=sess)
    u2t_model = user2track_model(num_inputs, num_outputs, neg_k, embedding, track_index, lr)
    
    def exp_decay(epoch):
        initial_lrate = lr
        k = 0.2
        lrate = initial_lrate * np.exp(-k * epoch)
        print("Epoch {} : Learning rate {}".format(epoch, lrate))
        return lrate
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='cosine_proximity', patience=1,
                                         restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(exp_decay),
        ]
    
    return u2t_model, train_generator, validation_generator, callbacks