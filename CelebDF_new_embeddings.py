import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from os.path import join
from pathlib import Path
from collections import Counter
from warnings import simplefilter

from utils import make_short_file


def clean_raw_csv(infile, outfile):
    """
        Files: one_segment, two_segments
        Columns: image_path, filename, folder, target, logit_1, logit_2, feature_embedding x 768

        File: test_embeddings
        Columns: image_path,filename,folder,run_type,target,logit_1,logit_2,e000...e767
    """
    # save_subset_of_csv()
    columns = ['image_path', 'filename', 'folder', 'run_type', 'target', 'logit_1', 'logit_2']
    for i in range(768):
        columns.append(f'e{i:03d}')

    import csv
    with open(infile, encoding='utf-8') as f, open(outfile, 'w', newline='') as o:
        reader = csv.reader(f)
        writer = csv.writer(o, delimiter=',')  # adjust as necessary
        writer.writerow(columns)
        for r_row in reader:
            if len(r_row) == 8:
                continue
            writer.writerow(r_row)
    print('Done')


def save_predictions_only(infile, outfile):
    df_chunks = pd.read_csv(infile,
                            chunksize=10000,
                            low_memory=False,
                            index_col=False,
                            header=0)
    df = pd.concat(df_chunks)
    df = df[['filename', 'folder', 'target', 'logit_1', 'logit_2']]
    df.to_csv(outfile, index=False)


def evaluate_video_and_frame_level(infile):
    df = pd.read_csv(infile, index_col=False)
    grouped_df = df.groupby('folder')  # folder == video_name
    predictions, ground_truths = [], []
    results_log, wrong_predictions = [], []
    frame_lvl_acc_list = []
    frame_lvl_preds, frame_lvl_gts = [], []
    for video_name, filtered_df in tqdm(grouped_df):
        logits = filtered_df[['logit_1', 'logit_2']].to_numpy()
        preds_list = list(np.argmax(logits, axis=1))  # frame level prediction
        final_pred = max(preds_list, key=preds_list.count)  # video level prediction
        _gt = filtered_df[filtered_df['folder'] == video_name].iloc[0]
        gt = _gt['target']

        # frame level metrics
        frame_lvl_acc_list.append(accuracy_score([gt]*len(preds_list), preds_list))
        frame_lvl_preds.extend(preds_list)
        frame_lvl_gts.extend([gt]*len(preds_list))

        if final_pred != gt:
            wrong_predictions.append(video_name)

        predictions.append(final_pred)
        ground_truths.append(gt)
        results_log.append({
            'video_name': video_name,
            'prediction': final_pred,
            'ground_truth': gt
        })
    # pd.DataFrame(results_log).to_csv('./CelebDF_test_results_log_IN21K.csv')

    accuracy = sum(1 for _pred, _gt in zip(predictions, ground_truths) if _pred == _gt) / len(predictions)
    auc = roc_auc_score(ground_truths, predictions, average="macro")
    print(f'Video level Accuracy: {accuracy}. AUC: {auc}.\n')

    # frame level
    f_accuracy = sum(frame_lvl_acc_list) / len(frame_lvl_acc_list)
    f_auc = roc_auc_score(frame_lvl_gts, frame_lvl_preds, average="macro")
    print(f'Frame level Accuracy: {f_accuracy}. AUC: {f_auc}.\n')

    # with open(r'./CeleBDF_test_wrong_predictions.txt', 'w') as fp:
    #     for item in wrong_predictions:
    #         # write each item on a new line
    #         fp.write("%s\n" % item)
    #     print('Done: Wrong Predictions saved.')


if __name__ == '__main__':
    # make_short_file(infile=r'./data/CelebDF_test_embeddings_cleaned.csv',
    #                 outfile=r'./data/CelebDF_test_embeddings_cleaned_short.csv',
    #                 use_pandas=False,
    #                 sample_size=1000)

    # clean_raw_csv(infile=r'./data/CelebDF_test_embeddings.csv',
    #               outfile=r'./data/CelebDF_test_embeddings_cleaned.csv')

    # save_predictions_only(infile=r'./data/CelebDF_test_embeddings_cleaned.csv',
    #                       outfile=r'./data/CelebDF_test_predictions.csv')

    evaluate_video_and_frame_level(infile=r'./data/CelebDF_test_predictions.csv')