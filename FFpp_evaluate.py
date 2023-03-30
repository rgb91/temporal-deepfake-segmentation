import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from utils import load_data_single_npy, multiclass_roc_auc_score, remove_extension, calculate_IOU
from sklearn.metrics import roc_auc_score, accuracy_score
from warnings import simplefilter
from tensorflow.keras.models import load_model

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


def smooth_prediction(preds, n):
    updated_pred = []
    for i in range(0, len(preds), n):
        pred_chunk = preds[i:i + n]
        majority = max(pred_chunk, key=pred_chunk.count)
        updated_pred += [majority] * len(pred_chunk)
        # print(i, majority, pred_chunk)
    return updated_pred


def smooth_prediction_v2(preds, offset=3):
    updated_preds = []
    for i, pred in enumerate(preds):
        ll = max(0, i - offset)  # left offset
        # preds_l = preds[ll:i]
        preds_l = updated_preds[ll:i]
        preds_r = preds[i + 1:i + offset+1]

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


def evaluate_old():
    """
    x_test      (1273, 500, 768)
    y_test      (1273,)
    y_pred_raw  (1273, 6)
    y_pred      (1273,)
    """
    data_path = '/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings/DELETE_FFpp_embeddings_Method_A' \
                '/FFpp_embeddings_Method_A_val.npy'
    x_test, y_test = load_data_single_npy(data_path)
    y_test = y_test[:, 0]
    # print(Counter(y_test))

    model = keras.models.load_model(MODEL_PATH)
    y_pred_raw = model.predict(x_test)

    y_pred = np.argmax(y_pred_raw, axis=1)
    print(x_test.shape)
    print(y_test.shape)
    print(y_pred_raw.shape)
    print(y_pred.shape)
    # exit(1)
    # print(Counter(y_pred))

    # auc = roc_auc_score(y_test, y_pred, average="macro")
    auc_macro = multiclass_roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # print(f'AUC: {auc:0.4f}')
    print(f'AUC (macro): {auc_macro:0.4f}')
    print(f'Accuracy: {acc:0.4f}')


def evaluate_test_data_ViT(raw_csv_file='./data/FFpp_test_predictions.csv'):
    df_all_chunks = pd.read_csv(raw_csv_file, index_col=False, chunksize=10000)
    df = pd.concat(df_all_chunks)
    print('\n\nREADING DONE.\n\n')

    df.loc[:, 'filename'] = df['filename'].str.replace('.jpg', '')
    df.loc[:, 'filename'] = pd.to_numeric(df['filename'])

    grouped_df = df.groupby('folder')  # folder == video_name
    predictions, ground_truths = [], []
    results_log, wrong_predictions = [], []
    for video_name, filtered_df in tqdm(grouped_df):
        filtered_df_sorted = filtered_df.sort_values('filename')
        logits = filtered_df_sorted[['logit_1', 'logit_2']].to_numpy()
        preds_list = list(np.argmax(logits, axis=1))

        final_pred = max(preds_list, key=preds_list.count)
        gt = filtered_df_sorted[filtered_df_sorted['folder'] == video_name].iloc[0]

        if final_pred != gt['target']:
            wrong_predictions.append(video_name)

        predictions.append(final_pred)
        ground_truths.append(gt['target'])
        results_log.append({
            'video_name': video_name,
            'prediction': final_pred,
            'ground_truth': gt
        })
    pd.DataFrame(results_log).to_csv('./FFpp_test_results_log_IN21K.csv')

    accuracy = sum(1 for _pred, _gt in zip(predictions, ground_truths) if _pred == _gt) / len(predictions)
    auc = roc_auc_score(ground_truths, predictions, average="macro")
    print(f'Video level Accuracy: {accuracy}. AUC: {auc}')

    with open(r'./FFpp_test_wrong_predictions.txt', 'w') as fp:
        for item in wrong_predictions:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done: Wrong Predictions saved.')


def evaluate_temporal_one_segment_ViT(results_csv_file='./results_one_segment_vit_wo_smoothing.csv', smooth_n_frames=0):
    """

    Variation 1: Only frame level accuracy, ViT model, Acc, IOU, No smoothing

    prediction: 1 => real
    prediction: 0 => fake
    :return:
    """
    real, fake = 1, 0
    data_path = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings/FFpp_temporal_eval_data/one_segment'
    gt_file = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings/FFpp_temporal_eval_data' \
              r'/temporal_one_segment_ground_truth.csv'
    gt_df = pd.read_csv(gt_file, index_col=False)

    datasets = ['fake_DF', 'fake_FSh', 'fake_F2F', 'fake_NT', 'fake_FS']
    acc_total, IoU_total = 0, 0
    predictions, ground_truths = [], []
    for d in datasets:
        d_path = os.path.join(data_path, d)
        acc_one_dataset_total, IoU_one_dataset_total = 0, 0
        for v_name in os.listdir(d_path):
            v_preds_df = pd.read_csv(os.path.join(d_path, v_name), index_col=False)
            v_preds = v_preds_df['prediction'].tolist()
            if smooth_n_frames > 0:
                v_preds = smooth_prediction_v2(v_preds, smooth_n_frames)

            gt_data = gt_df[gt_df['video_name'] == remove_extension(v_name)].iloc[0]
            total_frames, fake_start, fake_end = max(gt_data['total_frames'], len(v_preds)), gt_data['fake_start'], \
                gt_data['fake_end']
            gt_labels = [real] * (fake_start) + [fake] * (fake_end - fake_start) + [real] * (total_frames - fake_end)

            predictions.extend(v_preds)
            ground_truths.extend(gt_labels)

            if len(gt_labels) != len(v_preds):
                # logging.info(f'GT and Predictions length mismatch. Dataset: {d}, Video: {remove_extension(v_name)}.\n'
                #              f'Length of GT: {len(gt_labels)}, Length of Predictions: {len(v_preds)}')
                continue

            acc_one_vid = sum(1 for gt, pred in zip(gt_labels, v_preds) if gt == pred) / len(gt_labels)  # frame level
            IoU_one_vid = calculate_IOU(gt_labels, v_preds)
            # result_one_vid = pd.DataFrame([{'dataset': d,
            #                                 'video_name': v_name,
            #                                 'acc': acc_one_vid,
            #                                 'IOU': IoU_one_vid}])
            # results_log = pd.concat([results_log, result_one_vid], ignore_index=True)

            acc_one_dataset_total += acc_one_vid
            IoU_one_dataset_total += IoU_one_vid

        acc_one_dataset_avg = acc_one_dataset_total / len(os.listdir(d_path))
        IoU_one_dataset_avg = IoU_one_dataset_total / len(os.listdir(d_path))
        print(f'Dataset {d}: \n\tAccuracy: {acc_one_dataset_avg:0.4f}\n\tIOU: {IoU_one_dataset_avg:0.4f}')

        acc_total += acc_one_dataset_avg
        IoU_total += IoU_one_dataset_avg

    acc_total_avg = acc_total / len(datasets)
    IoU_total_avg = IoU_total / len(datasets)
    auc_total_avg = roc_auc_score(ground_truths, predictions, average="macro")
    print(f'\nTotal: \n\tAccuracy: {acc_total_avg:0.4f}\n\tIoU: {IoU_total_avg:0.4f}')
    print(f'\tAUC: {auc_total_avg:0.4f}')
    # results_log.to_csv(results_csv_file, index=False)


def evaluate_temporal_two_segments_ViT(results_csv_file='./results_two_segments_vit_wo_smoothing.csv', smooth_n_frames=0):
    """
    Variation 1: Only frame level accuracy, ViT model, Acc, IOU, No smoothing

    prediction: 1 => real
    prediction: 0 => fake
    :return:
    """
    results_log = pd.DataFrame(columns=['dataset', 'video_name', 'acc', 'IOU'])
    data_path = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings/FFpp_temporal_eval_data/two_segments'

    real, fake = 1, 0
    gt_file = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings/FFpp_temporal_eval_data' \
              r'/temporal_two_segments_ground_truth.csv'
    gt_df = pd.read_csv(gt_file, index_col=False)

    datasets = ['fake_DF', 'fake_FSh', 'fake_F2F', 'fake_NT', 'fake_FS']
    acc_total, IOU_total = 0, 0
    predictions, ground_truths = [], []
    for d in datasets:
        d_path = os.path.join(data_path, d)
        acc_one_dataset_total, IOU_one_dataset_total = 0, 0
        for v_name in os.listdir(d_path):
            v_preds_df = pd.read_csv(os.path.join(d_path, v_name), index_col=False)
            v_preds = v_preds_df['prediction'].tolist()
            if smooth_n_frames > 0:
                v_preds = smooth_prediction_v2(v_preds, smooth_n_frames)

            gt_data = gt_df[gt_df['video_name'] == remove_extension(v_name)].iloc[0]
            total_frames = max(gt_data['total_frames'], len(v_preds))
            fake1_start, fake1_end = gt_data['fake1_start'], gt_data['fake1_end']
            fake2_start, fake2_end = gt_data['fake2_start'], gt_data['fake2_end']
            gt_labels = [real] * (fake1_start) + \
                        [fake] * (fake1_end - fake1_start) + \
                        [real] * (fake2_start - fake1_end) + \
                        [fake] * (fake2_end - fake2_start) + \
                        [real] * (total_frames - fake2_end)

            predictions.extend(v_preds)
            ground_truths.extend(gt_labels)

            if len(gt_labels) != len(v_preds):
                # logging.info(f'GT and Predictions length mismatch. Dataset: {d}, Video: {remove_extension(v_name)}.\n'
                #              f'Length of GT: {len(gt_labels)}, Length of Predictions: {len(v_preds)}')
                continue

            acc_one_vid = sum(1 for gt, pred in zip(gt_labels, v_preds) if gt == pred) / len(gt_labels)  # frame level
            IOU_one_vid = calculate_IOU(gt_labels, v_preds)
            result_one_vid = pd.DataFrame([{'dataset': d,
                                            'video_name': v_name,
                                            'acc': acc_one_vid,
                                            'IOU': IOU_one_vid}])
            results_log = pd.concat([results_log, result_one_vid], ignore_index=True)

            acc_one_dataset_total += acc_one_vid
            IOU_one_dataset_total += IOU_one_vid
        acc_one_dataset_avg = acc_one_dataset_total / len(os.listdir(d_path))
        IOU_one_dataset_avg = IOU_one_dataset_total / len(os.listdir(d_path))
        print(f'Dataset {d}: \n\tAccuracy: {acc_one_dataset_avg:0.4f}\n\tIOU: {IOU_one_dataset_avg:0.4f}')

        acc_total += acc_one_dataset_avg
        IOU_total += IOU_one_dataset_avg

    acc_total_avg = acc_total / len(datasets)
    IoU_total_avg = IOU_total / len(datasets)
    auc_total_avg = roc_auc_score(ground_truths, predictions, average="macro")
    print(f'\nTotal: \n\tAccuracy: {acc_total_avg:0.4f}\n\tIOU: {IoU_total_avg:0.4f}')
    print(f'\tAUC: {auc_total_avg:0.4f}')

    # results_log.to_csv(results_csv_file, index=False)


def evaluate_video_level_timeseries(in_dir):
    """
    real => 0
    fake => 1, 2, 3, 4, 5 (multi-class)
    fake => 1 (binary)
    """
    model = load_model(MODEL_PATH)

    predictions, ground_truths = [], []
    wrong_predictions, results_log = [], []
    all_predictions, all_ground_truths = [], []
    for video_name_npy in tqdm(os.listdir(in_dir)):
        data = np.load(os.path.join(in_dir, video_name_npy))
        x_test, y_test_list = data[:, :, 1:], data[:, 0, 0]
        y_test_list = y_test_list.tolist()

        y_pred_raw = model.predict(x_test)
        # y_pred_list = y_pred_raw[:, 0].tolist()
        y_pred_list = list(np.argmax(y_pred_raw, axis=1))

        y_pred = max(y_pred_list, key=y_pred_list.count)
        y_test = max(y_test_list, key=y_test_list.count)

        # y_pred = 1 if y_pred >= 1 else 0  # TODO temporary, remove
        # y_test = 1 if y_test >= 1 else 0  # TODO temporary, remove

        all_predictions.extend(y_pred_list)
        all_ground_truths.extend(y_test_list)

        predictions.append(y_pred)
        ground_truths.append(y_test)

        results_log.append({
            'video_name': video_name_npy,
            'prediction': y_pred,
            'ground_truth': y_test
        })

        # print(x_test.shape)
        # print('y test', y_test.shape)
        # print('pred raw', y_pred_raw.shape)
        # print('pred clean', y_pred.shape)

        # print(y_test[0], y_pred[0])
        # exit(1)
        # print(y_pred.shape)
        # print(Counter(y_pred))
        if y_pred != y_test:
            wrong_predictions.append(video_name_npy)

    # all_predictions = [1 if val >= 1 else 0 for val in all_predictions]  # TODO temporary, remove
    # all_ground_truths = [1 if val >= 1 else 0 for val in all_ground_truths]  # TODO temporary, remove

    acc = accuracy_score(ground_truths, predictions)
    auc = roc_auc_score(ground_truths, predictions, average="macro")
    print(f'Video level Accuracy: {acc:0.4f}.')
    print(f'Video level AUC (macro): {auc:0.4f}.\n')

    all_acc = accuracy_score(all_ground_truths, all_predictions)
    all_auc = roc_auc_score(all_ground_truths, all_predictions, average="macro")

    print(f'Window/Time-step level Accuracy: {all_acc:0.4f}.')
    print(f'Window/Time-step level AUC (macro): {all_auc:0.4f}.\n')

    print(f'Number of wrong predictions: {len(wrong_predictions)}.')

    # with open(r'FFpp_test_video_level_wrong_predictions_binary_Run14.txt', 'w') as fp:
    #     for item in wrong_predictions:
    #         fp.write("%s\n" % item)
    #     print('Done: Wrong Predictions saved.')


def evaluate_temporal_one_segment_timeseries(in_dir, smooth_n_frames=0):
    """
    real => 0
    fake => 1 (binary)
    """
    model = load_model(MODEL_PATH)

    datasets = ['fake_DF', 'fake_FSh', 'fake_F2F', 'fake_NT', 'fake_FS']
    predictions_per_dataset = {'fake_DF': [], 'fake_FSh': [], 'fake_F2F': [], 'fake_NT': [], 'fake_FS': []}
    ground_truths_per_dataset = {'fake_DF': [], 'fake_FSh': [], 'fake_F2F': [], 'fake_NT': [], 'fake_FS': []}
    predictions, ground_truths = [], []
    for video_name_npy in tqdm(os.listdir(in_dir)):
        dataset = video_name_npy.split('_')[0] + '_' + video_name_npy.split('_')[1]

        data = np.load(os.path.join(in_dir, video_name_npy))
        x_test, y_test = data[:, :, 1:], data[:, 0, 0]

        y_preds_raw = model.predict(x_test)
        y_preds = np.argmax(y_preds_raw, axis=1)
        if smooth_n_frames > 0:
            y_preds = smooth_prediction_v2(list(y_preds), smooth_n_frames)
            # y_preds = smooth_prediction(list(y_preds), smooth_n_frames)

        predictions.extend(list(y_preds))
        ground_truths.extend(list(y_test))
        predictions.extend([1, 1, 1, 1])  # TODO remove (only for 5-timestep models' missing frames)
        ground_truths.extend([1, 1, 1, 1])  # TODO remove (only for 5-timestep models' missing frames)

        predictions_per_dataset[dataset].extend(list(y_preds))
        ground_truths_per_dataset[dataset].extend(list(y_test))

    for d in datasets:
        d_pred = predictions_per_dataset[d]
        d_gt = ground_truths_per_dataset[d]
        d_auc_macro = roc_auc_score(d_gt, d_pred, average="macro")
        d_acc = accuracy_score(d_gt, d_pred)
        d_iou = calculate_IOU(d_gt, d_pred)
        print(f'Dataset: {d}')
        print(f'\tAccuracy: {d_acc:0.4f}.')
        print(f'\tIoU: {d_iou:0.4f}.')
        print(f'\tAUC: {d_auc_macro:0.4f}.\n')

    auc_macro = roc_auc_score(ground_truths, predictions, average="macro")
    acc = accuracy_score(ground_truths, predictions)
    iou = calculate_IOU(ground_truths, predictions)
    print('Average:')
    print(f'\tAccuracy: {acc:0.4f}.')
    print(f'\tIoU: {iou:0.4f}.')
    print(f'\tAUC: {auc_macro:0.4f}.\n')


def evaluate_temporal_two_segments_timeseries(in_dir):
    pass


if __name__ == '__main__':
    DATA_ROOT_FFpp = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings'
    DATA_ROOT_CelebDF = r'/data/PROJECT FILES/DFD_Embeddings/CelebDF_embeddings'
    MODEL_PATH = r'./saved_models/FFpp_TimeSeries_Run22_best.h5'
    # MODEL_PATH = r'./saved_models/FFpp_Baseline_Run01_best.h5'

    # evaluate_old()

    # evaluate_test_data_ViT(r'./data/FFpp_test_predictions.csv')

    # evaluate_temporal_one_segment_ViT(smooth_n_frames=5)

    # evaluate_temporal_two_segments_ViT(smooth_n_frames=5)

    evaluate_video_level_timeseries(in_dir=os.path.join(DATA_ROOT_FFpp,
                                                        'FFpp_test_embeddings_npy_videos_10steps_binary_overlap'))

    # evaluate_temporal_one_segment_timeseries(in_dir=os.path.join(DATA_ROOT_FFpp, 'FFpp_temporal_one_segment_npy_5steps_overlap'),
    #                                          smooth_n_frames=25)

    # evaluate_temporal_two_segments_timeseries(in_dir=os.path.join(DATA_ROOT_FFpp, 'FFpp_temporal_one_segment_npy_5steps_overlap'),
    #                                          smooth_n_frames=25)