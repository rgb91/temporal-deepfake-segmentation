import os
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils import data_reader_from_npy
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

MODEL_PATH = r'./saved_models/FFpp_Method_A_TimeSeries_Run02/'
DATA_PATH = 'data/FFpp_embeddings_Method_A/FFpp_embeddings_Method_A_val.npy'


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)


def main():
    x_test, y_test = data_reader_from_npy(DATA_PATH)
    y_test = y_test[:, 0]
    # print(y_test.shape)
    # print(Counter(y_test))

    model = keras.models.load_model(MODEL_PATH)
    y_pred_raw = model.predict(x_test)
    # print(y_pred_raw.shape)

    y_pred = np.argmax(y_pred_raw, axis=1)
    # print(y_pred.shape)
    # print(Counter(y_pred))

    # auc = roc_auc_score(y_test, y_pred, average="macro")
    auc_macro = multiclass_roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # print(f'AUC: {auc:0.4f}')
    print(f'AUC (macro): {auc_macro:0.4f}')
    print(f'Accuracy: {acc:0.4f}')


if __name__ == '__main__':
    main()


"""
Run02

Counter({0: 273, 3: 200, 1: 200, 5: 200, 4: 200, 2: 200})
Counter({0: 267, 2: 219, 1: 202, 3: 198, 5: 195, 4: 192})
AUC (macro): 0.9798
Accuracy: 0.9647
"""