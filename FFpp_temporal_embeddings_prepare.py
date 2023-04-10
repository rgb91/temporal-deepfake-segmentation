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


def group_by_video_FFpp_temporal_predictions_only(infile, out_dir, n_segments='one_segment'):
    datasets = ['fake_DF', 'fake_F2F', 'fake_FS', 'fake_FSh', 'fake_NT']
    # df_all = pd.read_csv(infile, index_col=False)
    df_all_chunks = pd.read_csv(infile, index_col=False, chunksize=10000)
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

            Path(join(out_dir, d)).mkdir(parents=True, exist_ok=True)
            filtered_df_d_sorted.to_csv(join(out_dir, d, video_name + '.csv'), index=False)
            # exit(1)
            print(f'Dataset {d} COMPlETE.\n')
    print('DONE.\n')
    pass


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
    group_by_video_FFpp_temporal_predictions_only(infile=join(DATA_ROOT, f'FFpp_temporal_{n_segments}_embeddings_clean.csv'),
                                                  out_dir=join(DATA_ROOT, 'videos_preds', n_segments),
                                                  n_segments=n_segments)

    n_frames = 5
    # eval_two_segments_ViT(results_csv_file=fr'./results_two_segments_vit_wo_smoothing_{n_frames}_frames.csv',
    #                  smooth_n_frames=n_frames)

    # clean_raw_csv(r'./data/FFpp_test_embeddings.csv', r'./data/FFpp_test_embeddings_cleaned_3.csv')