import tensorflow as tf
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_play, x_skip, y_play, track_index, freq_prob, neg_k=10, batch_size=512, num_inputs=100,
                 shuffle=True):
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.x_play = x_play
        self.x_skip = x_skip
        self.y_play = y_play
        self.track_index = track_index
        self.freq_prob = freq_prob
        self.neg_k = neg_k
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.x_play) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_play, x_skip, pos_input, neg_input, outputs = self.__data_generation(indexes)
        # Naming should be exactly same as you define on the layer graph
        return [x_play, x_skip, pos_input, neg_input], [outputs]
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x_play))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, index):
        y_neg = np.random.choice(len(self.track_index.index),
                                 (self.y_play[index].shape[0], self.neg_k * self.num_inputs),
                                 p=self.freq_prob)
        one_maskings = (self.y_play[index] > 0).astype(np.float32)
        zero_maskings = np.zeros((self.y_play[index].shape[0], self.neg_k * self.num_inputs), dtype=np.float32)
        for i, (pos, neg) in enumerate(zip(self.y_play[index], y_neg)):
            tmp_arr = zero_maskings[i]
            pos_nz = pos[pos > 0]
            for nz in pos_nz:
                tmp_arr += (neg == nz).astype(np.float32)
            zero_maskings[i] = tmp_arr.astype(np.float32)
        labels_data = np.concatenate((one_maskings, zero_maskings), axis=1)
        labels_data_ = np.where(labels_data > 0, labels_data, -1.0)
        return self.x_play[index], self.x_skip[index], self.y_play[index], y_neg, labels_data_