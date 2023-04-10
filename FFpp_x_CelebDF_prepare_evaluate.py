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

from utils import make_short_file, split_dataframe


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


def get_class_name_from_video_name(video_name):
    if 'real' in video_name:
        return 'real'
    return 'synthesis'


def make_npy_by_video_FFpp_x_CelebDF(infile, out_dir, timesteps=5, overlap=0):
    class_name_num_map = {'real': 0, 'synthesis': 1}

    df_chunks = pd.read_csv(infile, index_col=False, chunksize=10000)
    df = pd.concat(df_chunks)
    print('\nDATA READING DONE.\n\n')

    df = df.drop(columns=['image_path', 'run_type', 'target', 'logit_1', 'logit_2'])
    grouped_df = df.groupby('folder')  # folder == video_name

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for video_name, filtered_df in tqdm(grouped_df):
        df_f = filtered_df.copy()  # apologies shortened name
        df_f['filename'] = df_f['filename'].str.replace('.jpg', '')
        try:
            df_f['filename'] = pd.to_numeric(df_f['filename'])
        except ValueError:
            continue

        df_f = df_f.sort_values('filename')
        df_f = df_f.drop(columns=['folder'])

        df_f = df_f.drop_duplicates(subset=['filename'])
        # print(df_f.head())
        # df_f.to_csv(os.path.join(out_dir, video_name + '.csv'), index=False)
        # exit(1)

        class_name = get_class_name_from_video_name(video_name)
        class_num = class_name_num_map[class_name]

        if overlap > 0:
            df_splits = [df_f.iloc[i:i + timesteps, :]
                         for i in range(0, len(df_f), timesteps - overlap) if i < len(df_f) - timesteps]
        else:
            df_splits = split_dataframe(df_f, chunk_size=timesteps)

        data_np = None
        for df_split in df_splits:
            df_split = df_split.drop(columns=['filename'])  # remove sequence number
            df_split.insert(0, 'class', int(class_num))  # set class number
            df_np = df_split.to_numpy()
            df_np = np.expand_dims(df_np, axis=0)  # 1 x n_timesteps x embedding_length

            if overlap < (timesteps-1):  # if overlap == timesteps-1 then now need to add padding
                df_np = np.pad(df_np, ((0, 0), (0, timesteps - df_np.shape[1]), (0, 0)), 'constant', constant_values=0)

            if data_np is None:
                data_np = df_np
            else:
                data_np = np.vstack([data_np, df_np])
        np.save(f'{out_dir}/{video_name}.npy', data_np)


def evaluate_ViT(infile):
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
    DATA_ROOT = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings/cross/CelebDF'

    # clean_raw_csv(infile=r'./data/CelebDF_test_embeddings.csv',
    #               outfile=r'./data/CelebDF_test_embeddings_cleaned.csv')

    # make_short_file(infile=r'./data/CelebDF_test_embeddings_cleaned.csv',
    #                 outfile=r'./data/CelebDF_test_embeddings_cleaned_short.csv',
    #                 use_pandas=False,
    #                 sample_size=1000)

    # save_predictions_only(infile=r'./data/CelebDF_test_embeddings_cleaned.csv',
    #                       outfile=r'./data/CelebDF_test_predictions.csv')

    # evaluate_ViT(infile=r'./data/CelebDF_test_predictions.csv')

    make_npy_by_video_FFpp_x_CelebDF(infile=os.path.join(DATA_ROOT, 'FFpp_x_CelebDF_test_embeddings_cleaned.csv'),
                                     out_dir=os.path.join(DATA_ROOT, 'FFpp_x_CelebDF_npy_videos_5steps_overlap'),
                                     timesteps=5,
                                     overlap=4)