import numpy as np
import keras.backend as K
import math


def F1_score(y_true, y_pred):
    # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def readucr(filepath):
    data = np.loadtxt(filepath, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def data_reader_from_npy(filepath):
    data = np.load(filepath)
    x = data[:, :, 1:]
    y = data[:, :, 0]
    return x, y


def print_dict(dictionary):
    for k, v in dictionary.items():
        print(f'{k}: {v}')
    print()


def check_np_padding():
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(a)
    print(a.shape)
    a_pad = np.pad(a, ((0, 0), (0, 4), (0, 0)), 'constant', constant_values=0)
    print(a_pad)
    print(a_pad.shape)


def split_dataframe(df, chunk_size=500):
    chunks = list()
    num_chunks = math.ceil(len(df) / chunk_size)
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks
