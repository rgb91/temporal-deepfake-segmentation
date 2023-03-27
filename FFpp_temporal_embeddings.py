import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from os.path import join
from pathlib import Path
from collections import Counter
from warnings import simplefilter


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


def remove_extension(filename):
    return filename.split('.')[0]


def calculate_IOU(gt_labels, preds):
    intersection = sum(1 for gt, pred in zip(gt_labels, preds) if gt == pred)
    union = intersection + (len(gt_labels) - intersection) * 2
    return intersection / union


def smooth_prediction(preds, n):
    updated_pred = []
    for i in range(0, len(preds), n):
        pred_chunk = preds[i:i + n]
        majority = max(pred_chunk, key=pred_chunk.count)
        updated_pred += [majority] * len(pred_chunk)
        # print(i, majority, pred_chunk)
    return updated_pred


def remove_duplicate_rows_and_save():
    # Get count duplicate rows
    filepath = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_temporal/FFpp_temporal_one_segment_embeddings_2.csv'
    chunks = pd.read_csv(filepath, chunksize=10000)
    df = pd.concat(chunks)
    df_no_dup = df.drop_duplicates(keep='first')
    df_no_dup.to_csv(r'/data/PROJECT FILES/DFD_Embeddings/FFpp_temporal/FFpp_temporal_one_segment_embeddings_3.csv')
    # num_of_duplicate_rows = len(df) - len(df.drop_duplicates())
    # print(num_of_duplicate_rows)


def clean_raw_csv_FFpp_temporal(infile, outfile):
    """
        Files: one_segment, two_segments
        Columns: image_path, filename, folder, target, logit_1, logit_2, feature_embedding x 768

        File: test_embeddings
        Columns: image_path,filename,folder,run_type,target,logit_1,logit_2,e000...e767
    """
    columns = ['image_path', 'filename', 'folder', 'run_type', 'target', 'logit_1', 'logit_2']
    for i in range(768):
        columns.append(f'e{i:03d}')

    import csv

    first_row = True
    with open(infile, encoding='utf-8') as f, open(outfile, 'w') as o:
        reader = csv.reader(f)
        writer = csv.writer(o, delimiter=',')  # adjust as necessary
        writer.writerow(columns)
        for r_row in reader:
            if first_row:
                w_row = [item for item in columns]
                first_row = False
            else:
                w_row = [item for item in r_row]
            writer.writerow(w_row)
    print('Done')


def group_by_video_FFpp_temporal(input_csv_path, output_directory, n_segments='one_segment'):
    datasets = ['fake_DF', 'fake_F2F', 'fake_FS', 'fake_FSh', 'fake_NT']
    # df_all = pd.read_csv(input_csv_file, index_col=False)
    df_all_chunks = pd.read_csv(input_csv_path, index_col=False, chunksize=10000)
    df_all = pd.concat(df_all_chunks)
    print('Data Read Complete.\n')

    df = df_all.loc[:, ('image_path', 'filename', 'folder', 'logit_1', 'logit_2')]
    df['image_path'] = df['image_path'].str.replace(f'/data/gpfs/projects/punim1875/FFpp_temporal_{n_segments}/',
                                                    '')  # redundant, so remove
    df['dataset'] = df['image_path'].str.split('/').str[0]
    df.loc[:, 'filename'] = df['filename'].str.replace('.jpg', '')
    df.loc[:, 'filename'] = pd.to_numeric(df['filename'])
    df.drop(columns=['image_path'], inplace=True)

    grouped_df = df.groupby('folder')  # folder == video_name
    for video_name, filtered_df in grouped_df:
        for d in datasets:
            filtered_df_d = filtered_df[filtered_df['dataset'] == d]
            filtered_df_d_sorted = filtered_df_d.sort_values('filename')

            logits = filtered_df_d_sorted[['logit_1', 'logit_2']].to_numpy()
            preds = np.argmax(logits, axis=1)
            filtered_df_d_sorted.loc[:, 'prediction'] = preds.tolist()

            Path(join(output_directory, d)).mkdir(parents=True, exist_ok=True)
            filtered_df_d_sorted.to_csv(join(output_directory, d, video_name + '.csv'), index=False)
            # exit(1)
            print(f'Dataset {d} COMPlETE.\n')
    print('DONE.\n')
    pass


def eval_one_segment(results_csv_file='./results_one_segment_vit_wo_smoothing.csv', smooth_n_frames=0):
    """

    Variation 1: Only frame level accuracy, ViT model, Acc, IOU, No smoothing

    prediction: 1 => real
    prediction: 0 => fake
    :return:
    """
    results_log = pd.DataFrame(columns=['dataset', 'video_name', 'acc', 'IOU'])

    real, fake = 1, 0
    gt_file = r'data/FFpp_temporal_eval_data/temporal_one_segment_ground_truth.csv'
    gt_df = pd.read_csv(gt_file, index_col=False)

    datasets = ['fake_DF', 'fake_FSh', 'fake_F2F', 'fake_NT', 'fake_FS']
    acc_total, IOU_total = 0, 0
    for d in datasets:
        d_path = join(r'data/FFpp_temporal_eval_data/one_segment', d)
        acc_one_dataset_total, IOU_one_dataset_total = 0, 0
        for v_name in os.listdir(d_path):
            v_preds_df = pd.read_csv(join(d_path, v_name), index_col=False)
            v_preds = v_preds_df['prediction'].tolist()
            if smooth_n_frames > 0:
                v_preds = smooth_prediction(v_preds, smooth_n_frames)

            gt_data = gt_df[gt_df['video_name'] == remove_extension(v_name)].iloc[0]
            total_frames, fake_start, fake_end = max(gt_data['total_frames'], len(v_preds)), gt_data['fake_start'], \
            gt_data['fake_end']
            gt_labels = [real] * (fake_start) + [fake] * (fake_end - fake_start) + [real] * (total_frames - fake_end)

            if len(gt_labels) != len(v_preds):
                logging.info(f'GT and Predictions length mismatch. Dataset: {d}, Video: {remove_extension(v_name)}.\n'
                             f'Length of GT: {len(gt_labels)}, Length of Predictions: {len(v_preds)}')
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
        print(f'Dataset {d}: \n\tAccuracy: {acc_one_dataset_avg}\n\tIOU: {IOU_one_dataset_avg}')

        acc_total += acc_one_dataset_avg
        IOU_total += IOU_one_dataset_avg

    acc_total_avg = acc_total / len(datasets)
    IOU_total_avg = IOU_total / len(datasets)
    print(f'\nTotal: \n\tAccuracy: {acc_total_avg}\n\tIOU: {IOU_total_avg}')

    results_log.to_csv(results_csv_file, index=False)


def eval_two_segments(results_csv_file='./results_two_segments_vit_wo_smoothing.csv', smooth_n_frames=0):
    """
    Variation 1: Only frame level accuracy, ViT model, Acc, IOU, No smoothing

    prediction: 1 => real
    prediction: 0 => fake
    :return:
    """
    results_log = pd.DataFrame(columns=['dataset', 'video_name', 'acc', 'IOU'])

    real, fake = 1, 0
    gt_file = r'data/FFpp_temporal_eval_data/temporal_two_segments_ground_truth.csv'
    gt_df = pd.read_csv(gt_file, index_col=False)

    datasets = ['fake_DF', 'fake_FSh', 'fake_F2F', 'fake_NT', 'fake_FS']
    acc_total, IOU_total = 0, 0
    for d in datasets:
        d_path = join(r'data/FFpp_temporal_eval_data/two_segments', d)
        acc_one_dataset_total, IOU_one_dataset_total = 0, 0
        for v_name in os.listdir(d_path):
            v_preds_df = pd.read_csv(join(d_path, v_name), index_col=False)
            v_preds = v_preds_df['prediction'].tolist()
            if smooth_n_frames > 0:
                v_preds = smooth_prediction(v_preds, smooth_n_frames)

            gt_data = gt_df[gt_df['video_name'] == remove_extension(v_name)].iloc[0]
            total_frames = max(gt_data['total_frames'], len(v_preds))
            fake1_start, fake1_end = gt_data['fake1_start'], gt_data['fake1_end']
            fake2_start, fake2_end = gt_data['fake2_start'], gt_data['fake2_end']
            gt_labels = [real] * (fake1_start) + \
                        [fake] * (fake1_end - fake1_start) + \
                        [real] * (fake2_start - fake1_end) + \
                        [fake] * (fake2_end - fake2_start) + \
                        [real] * (total_frames - fake2_end)

            if len(gt_labels) != len(v_preds):
                logging.info(f'GT and Predictions length mismatch. Dataset: {d}, Video: {remove_extension(v_name)}.\n'
                             f'Length of GT: {len(gt_labels)}, Length of Predictions: {len(v_preds)}')
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
        print(f'Dataset {d}: \n\tAccuracy: {acc_one_dataset_avg}\n\tIOU: {IOU_one_dataset_avg}')

        acc_total += acc_one_dataset_avg
        IOU_total += IOU_one_dataset_avg

    acc_total_avg = acc_total / len(datasets)
    IOU_total_avg = IOU_total / len(datasets)
    print(f'\nTotal: \n\tAccuracy: {acc_total_avg}\n\tIOU: {IOU_total_avg}')

    results_log.to_csv(results_csv_file, index=False)


if __name__ == '__main__':
    DATA_ROOT = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings'
    # DATA_ROOT = r'/mnt/d/PROJECT FILES/DFD_Embeddings/FFpp_temporal'
    # DATA_ROOT = r'D:\PROJECT FILES\DFD_Embeddings\FFpp_temporal'

    # clean_raw_csv(infile=r'/data/PROJECT FILES/DFD_Embeddings/FFpp_temporal/test_embeddings.csv',
    #               outfile=r'/data/PROJECT FILES/DFD_Embeddings/FFpp_temporal/test_embeddings_clean.csv')

    # save_subset_of_csv(r'/data/PROJECT FILES/DFD_Embeddings/FFpp_temporal/test_embeddings_clean.csv',
    #                    r'/data/PROJECT FILES/DFD_Embeddings/FFpp_temporal/test_embeddings_part.csv',
    #                    sample_size=2000)

    n_segments = 'two_segments'
    # group_by_video_FFpp_temporal(input_csv_file=join(DATA_ROOT, f'FFpp_temporal_{n_segments}_embeddings_clean.csv'),
    #                output_directory=join(DATA_ROOT, 'videos_preds', n_segments),
    #                n_segments=n_segments)

    n_frames = 5
    # eval_two_segments(results_csv_file=fr'./results_two_segments_vit_wo_smoothing_{n_frames}_frames.csv',
    #                  smooth_n_frames=n_frames)

    # clean_raw_csv(r'./data/FFpp_test_embeddings.csv', r'./data/FFpp_test_embeddings_cleaned_3.csv')