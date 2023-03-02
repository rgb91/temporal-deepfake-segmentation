import os
import numpy as np
import pandas as pd
from warnings import simplefilter
from collections import Counter

from tqdm import tqdm

from utils import split_dataframe

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


def group_data_by_video_CelebDF(input_csv_path, output_directory):
    print(f'INPUT CSV: {input_csv_path}')
    df = pd.read_csv(input_csv_path, header=None)

    df = df.drop(columns=[0, 3])  # drop 'image_path' and 'set' columns after filtering is done

    df[1] = df[1].str.replace('.png', '')  # change filename to file-sequence-number
    df[1] = pd.to_numeric(df[1])

    grouped_df = df.groupby(2)
    for video_name, filtered_df in tqdm(grouped_df):
        if os.path.exists(os.path.join(output_directory, video_name + '.csv')):  # saved before, skip
            continue

        filtered_df = filtered_df.drop(columns=[2])  # drop folder_name/video_name
        filtered_df = filtered_df.sort_values(1)  # sort by sequence number
        # filtered_df = filtered_df.drop(columns=[1])  # drop sequence number

        filtered_df.to_csv(os.path.join(output_directory, video_name + '.csv'), header=False, index=False)


def make_npy_by_batch_WDF(vid_csv_dir, npy_out_dir, which_set='train', batch_size=64, timesteps=500):
    class_name_num_map = {'real': 0,
                          'fake': 1}

    data_np, i = None, 0
    for video_name_csv in tqdm(os.listdir(vid_csv_dir)):
        filepath = os.path.join(vid_csv_dir, video_name_csv)

        class_name = video_name_csv.split('_')[0]
        class_num = class_name_num_map[class_name]
        # print(class_name, video_name_csv)

        df = pd.read_csv(filepath, header=None)
        df_splits = split_dataframe(df, chunk_size=timesteps)
        # print(f'Number of splits: {len(df_splits)}')

        for df_split in df_splits:
            df_split = df_split.drop(columns=[0])  # remove sequence number
            df_split.insert(0, 'class', int(class_num))  # set class number
            df_np = df_split.to_numpy()
            df_np = np.expand_dims(df_np, axis=0)  # 1 x n_timesteps x embedding_length

            # padding to match n_timesteps for all videos, set to 500
            df_np = np.pad(df_np, ((0, 0), (0, timesteps-df_np.shape[1]), (0, 0)), 'constant', constant_values=0)

            if data_np is None:
                data_np = df_np
            else:
                data_np = np.vstack([data_np, df_np])

            if (i + 1) % batch_size == 0:
                np.save(f'{npy_out_dir}/WDF_embeddings_{which_set}_{(i + 1) // batch_size}.npy', data_np)
                data_np = None
            i += 1

    if data_np is not None:
        np.save(f'{npy_out_dir}/WDF_embeddings_{which_set}_{(i + 1) // batch_size}.npy', data_np)
    # print(data_np.shape)


def main():
    # group_data_by_video_CelebDF(
    #     r'./data/WDF_embeddings/train_embeddings.csv',
    #     r'./data/WDF_embeddings_train/'
    # )

    which_set = 'train'
    make_npy_by_batch_WDF(
        vid_csv_dir=fr'./data/WDF_embeddings_{which_set}/',
        npy_out_dir=fr'./data/WDF_embeddings_{which_set}_npy',
        which_set=which_set,
        timesteps=500,
        # method='B'
    )


if __name__ == '__main__':
    main()
