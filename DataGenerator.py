import os
import tensorflow as tf
import numpy as np
# from tensorflow import keras

from utils import load_data_single_npy


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data_path, which_set='', npy_prefix='', shuffle=True):
        """Initialization"""
        self.indexes = None
        self.set = which_set
        self.data_path = data_path
        self.shuffle = shuffle
        self.npy_prefix = npy_prefix
        self.list_ids = os.listdir(data_path)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_ids)

    def __getitem__(self, index):
        """
        'Generate one batch of data'
        """
        # Generate indexes of the batch
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        # x, y = self.__data_generation(list_ids_temp)
        if len(self.set) < 1:
            npy_filepath = os.path.join(self.data_path, f'{self.npy_prefix}_{index+1}.npy')
        else:
            npy_filepath = os.path.join(self.data_path, f'{self.npy_prefix}_{self.set}_{index + 1}.npy')
        x, y = load_data_single_npy(npy_filepath)
        y = y[:, 0]

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # def __data_generation(self, list_IDs_temp):
    #     """
    #     'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
    #     """
    #
    #
    #     # x = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     # y = np.empty(self.batch_size, dtype=int)
    #     #
    #     # # Generate data
    #     # for i, ID in enumerate(list_IDs_temp):
    #     #     # Store sample
    #     #     x[i, ] = np.load('data/' + ID + '.npy')
    #     #
    #     #     # Store class
    #     #     y[i] = self.labels[ID]
    #
    #     return x, y