import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
# import keras.backend as K
from tensorflow.keras import backend as K


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


def load_data_single_npy(filepath):
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


def load_data_multiple_npy(in_dir, filename_prefix):
    x, y = None, None
    n_files = len(os.listdir(in_dir))
    print(f'\nLoading Data: {in_dir}')

    for i in tqdm(range(1, n_files + 1)):
        npy_filepath = os.path.join(in_dir, f'{filename_prefix}_{i}.npy')
        x_temp, y_temp = load_data_single_npy(npy_filepath)
        if x is None and y is None:
            x, y = x_temp, y_temp
        else:
            x = np.vstack([x, x_temp])
            y = np.vstack([y, y_temp])
    y = y[:, 0]

    print(f'Data: {filename_prefix} | shapes (x, y): {x.shape}, {y.shape}\n')

    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]

    return x, y


def make_short_file(infile, outfile, use_pandas=True, sample_size=1000):
    if use_pandas:
        df_chunks = pd.read_csv(infile,
                                chunksize=10000,
                                low_memory=False,
                                index_col=False,
                                header=0)
        df = pd.concat(df_chunks)
        df.head(sample_size).to_csv(outfile, index=False)
    else:
        import csv
        with open(infile, encoding='utf-8') as f, open(outfile, 'w') as o:
            reader = csv.reader(f)
            writer = csv.writer(o, delimiter=',')  # adjust as necessary
            for i, r_row in enumerate(reader):
                if i == sample_size:
                    break
                w_row = [item for item in r_row]
                writer.writerow(w_row)
    print('Done')