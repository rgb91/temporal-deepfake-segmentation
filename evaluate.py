import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from os.path import join
from tensorflow import keras
from collections import Counter
from utils import load_data_single_npy, multiclass_roc_auc_score, remove_extension, calculate_IOU, get_position_encoding
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from warnings import simplefilter
from tensorflow.keras.models import load_model
import argparse

parser = argparse.ArgumentParser(description='FFpp Timeseries Training')
parser.add_argument('--model', help='Path to model', required=True)
parser.add_argument('--data', help='Path to data directory', required=True)
parser.add_argument('--variation', help='Options: subtle, random, video', required=True)
# parser.add_argument('--steps', help='Window size', required=True)
# parser.add_argument('--overlap', help='Overlap size of windows', required=True)
# parser.add_argument('--smooth', help='Smoothing window', default=15)

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


def smooth_predictions(preds, offset=3):
    updated_preds = []
    for i, pred in enumerate(preds):
        ll = max(0, i - offset)  # left offset
        # preds_l = preds[ll:i]
        preds_l = updated_preds[ll:i]
        preds_r = preds[i + 1:i + offset + 1]

        majority_l = max(preds_l, key=preds_l.count) if len(preds_l) > 0 else -1
        majority_r = max(preds_r, key=preds_r.count) if len(preds_r) > 0 else -1

        if majority_l < 0:  # left doesn't exist
            new_pred = majority_r if pred != majority_r else pred
        elif majority_r < 0:  # right doesn't exist
            new_pred = majority_l if pred != majority_l else pred
        else:
            if majority_l == majority_r:
                new_pred = majority_l if pred != majority_l else pred
            else:
                new_pred = pred
        updated_preds.append(new_pred)
    return updated_preds


def evaluate_video_level(in_dir, model_path, smooth_n_frames=25, dataset_level=True):
    """
    real => 0
    fake => 1, 2, 3, 4, 5 (multi-class)
    fake => 1 (binary)
    """
    model = load_model(model_path)

    predictions, ground_truths = [], []
    wrong_predictions, results_log, load_error_list = [], [], []

    datasets = ['DF', 'FSh', 'F2F', 'NT', 'FS', 'real']
    predictions_per_dataset = {'DF': [], 'FSh': [], 'F2F': [], 'NT': [], 'FS': [], 'real': []}
    ground_truths_per_dataset = {'DF': [], 'FSh': [], 'F2F': [], 'NT': [], 'FS': [], 'real': []}

    for video_name_npy in tqdm(os.listdir(in_dir)):
        try:
            data = np.load(join(in_dir, video_name_npy))
        except ValueError:
            load_error_list.append(video_name_npy)
            continue
        dataset = video_name_npy.split('_')[1] if video_name_npy.split('_')[0] == 'fake' else 'real'
        x_test, y_test_list = data[:, :, 1:], data[:, 0, 0]
        y_test_list = y_test_list.tolist()

        y_pred_raw = model.predict(x_test)
        y_pred_list = list(np.argmax(y_pred_raw, axis=1))
        if smooth_n_frames > 0:
            y_pred_list = smooth_predictions(list(y_pred_list), smooth_n_frames)

        y_preds = max(y_pred_list, key=y_pred_list.count)
        y_test = max(y_test_list, key=y_test_list.count)

        if dataset_level:
            predictions_per_dataset[dataset].append(y_preds)
            ground_truths_per_dataset[dataset].append(y_test)

        predictions.append(y_preds)
        ground_truths.append(y_test)

        if y_preds != y_test:
            wrong_predictions.append(video_name_npy)

    if dataset_level:
        for d in datasets:
            if d == 'real':
                continue
            d_pred = predictions_per_dataset[d] + predictions_per_dataset['real']
            d_gt = ground_truths_per_dataset[d] + ground_truths_per_dataset['real']
            d_auc_macro = roc_auc_score(d_gt, d_pred, average="macro")
            d_acc = accuracy_score(d_gt, d_pred)
            print(f'{d}, {d_acc:0.3f}, {d_auc_macro:0.3f}')

    # print(f'Number of errors in loading: {len(load_error_list)}.\n')

    acc = accuracy_score(ground_truths, predictions)
    auc = roc_auc_score(ground_truths, predictions, average="macro")
    print(f'Avg, {acc:0.3f}, {auc:0.3f}')


def evaluate_temporal(in_dir, model_path, smooth_n_frames=25, variation='subtle'):
    """
    Evaluate Temporal deepfakes
    real => 0, fake => 1 (binary)
    """

    if variation == 'subtle':
        datasets = ['F2F', 'NT']
        predictions_per_dataset = {'F2F': [], 'NT': []}
        ground_truths_per_dataset = {'F2F': [], 'NT': []}
    elif variation == 'random':
        datasets = ['fake_DF', 'fake_FSh', 'fake_F2F', 'fake_NT', 'fake_FS']
        predictions_per_dataset = {'fake_DF': [], 'fake_FSh': [], 'fake_F2F': [], 'fake_NT': [], 'fake_FS': []}
        ground_truths_per_dataset = {'fake_DF': [], 'fake_FSh': [], 'fake_F2F': [], 'fake_NT': [], 'fake_FS': []}
    else:
        print('Invalid argument `variation`.')
        exit(1)

    predictions, ground_truths = [], []
    model = load_model(model_path)

    for video_name_npy in tqdm(os.listdir(in_dir)):
        if variation == 'subtle':
            dataset = video_name_npy.split('_')[0]
        else:
            dataset = video_name_npy.split('_')[0] + '_' + video_name_npy.split('_')[1]

        data = np.load(join(in_dir, video_name_npy), allow_pickle=True)

        if variation == 'subtle':
            x_test, y_test = data[:, :, 1:769], data[:, 0, 0]
        else:
            x_test, y_test = data[:, :, 2:], data[:, 0, 0]
            x_test = x_test.astype(float)

        y_preds_raw = model.predict(x_test)
        y_preds = np.argmax(y_preds_raw, axis=1)
        if smooth_n_frames > 0:
            y_preds = smooth_predictions(list(y_preds), smooth_n_frames)

        predictions.extend(list(y_preds))
        ground_truths.extend(list(y_test))

        predictions_per_dataset[dataset].extend(list(y_preds))
        ground_truths_per_dataset[dataset].extend(list(y_test))

    print('Dataset, Accuracy, IoU, AUC')
    for d in datasets:
        d_pred = predictions_per_dataset[d]
        d_gt = ground_truths_per_dataset[d]
        d_auc = roc_auc_score(d_gt, d_pred, average="macro")
        d_acc = accuracy_score(d_gt, d_pred)
        d_iou = calculate_IOU(d_gt, d_pred)
        print(f'{d}, {d_acc:0.3f}, {d_iou:0.3f}, {d_auc:0.3f}')

    auc_macro = roc_auc_score(ground_truths, predictions, average="macro")
    acc = accuracy_score(ground_truths, predictions)
    iou = calculate_IOU(ground_truths, predictions)
    print(f'Avg, {acc:0.3f}, {iou:0.3f}, {auc_macro:0.3f}')


if __name__ == '__main__':
    args = vars(parser.parse_args())

    model_path = str(args['model'])
    data_dir = str(args['data'])
    variation = str(args['variation'])  # 'subtle' or 'random' or 'video'

    if not os.path.exists(model_path):
        print('Model does not exist.')
        exit(1)
    if not os.path.exists(data_dir):
        print('Data directory does not exist.')
        exit(1)
    if variation not in ['subtle', 'random', 'video']:
        print('Use \'subtle\' or \'random\' or \'video\' as variation.')
        exit(1)

    if variation == 'video':
        evaluate_video_level(
            in_dir=data_dir,
            model_path=model_path
        )
    else:
        evaluate_temporal(
            in_dir=data_dir,
            model_path=model_path,
            variation=variation
        )