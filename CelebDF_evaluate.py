import os
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from utils import data_reader_from_npy
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

MODEL_PATH = r'./saved_models/CelebDF_Method_B_TimeSeries_Run04/'
DATA_PATH = 'data/CelebDF_embeddings_Method_B_test_npy'


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)


def main():
    x_test, y_test = None, None
    n_files = len(os.listdir(DATA_PATH))
    for i in tqdm(range(1, n_files + 1)):
        npy_filepath = os.path.join(DATA_PATH, f'CelebDF_embeddings_Method_B_test_{i}.npy')
        x_temp, y_temp = data_reader_from_npy(npy_filepath)
        if x_test is None and y_test is None:
            x_test, y_test = x_temp, y_temp
        else:
            x_test = np.vstack([x_test, x_temp])
            y_test = np.vstack([y_test, y_temp])
    y_test = y_test[:, 0]
    print(y_test.shape)
    print(Counter(y_test))

    model = keras.models.load_model(MODEL_PATH)
    y_pred_raw = model.predict(x_test)
    print(y_pred_raw.shape)
    print(y_pred_raw[:10])

    # y_pred = np.argmax(y_pred_raw, axis=1)
    y_pred = [1*(y[0]>=0.5) for y in y_pred_raw]
    # print(y_pred.shape)
    print(y_pred[:10])
    print(Counter(y_pred))

    auc = roc_auc_score(y_test, y_pred, average="macro")
    auc_macro = multiclass_roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f'AUC: {auc:0.4f}')
    print(f'AUC (macro): {auc_macro:0.4f}')
    print(f'Accuracy: {acc:0.4f}')


if __name__ == '__main__':
    main()


"""
Run01

Counter({1.0: 1128, 0.0: 178})
Counter({1: 1306})

AUC (macro): 0.5000
Accuracy: 0.8637
"""