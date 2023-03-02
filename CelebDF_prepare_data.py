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


def check_np_padding():
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(a)
    print(a.shape)
    a_pad = np.pad(a, ((0, 0), (0, 4), (0, 0)), 'constant', constant_values=0)
    print(a_pad)
    print(a_pad.shape)


def group_data_by_video_CelebDF(input_csv_path, output_directory, which_set='val'):
    """
    From one BIG CSV file to many small CSV files, each small CSV file is for one video

    :param input_csv_path: path to the BIG CSV file
    :param output_directory: path to the directory of the small CSV files
    :param which_set: train or test
    :return: No return
    """
    print(f'INPUT CSV: {input_csv_path}')
    df = pd.read_csv(input_csv_path, header=None)

    df = df[df[3] == which_set]  # filter by 'set' (train or test)
    df = df.drop(columns=[2, 3])  # drop 'class label' and 'set' columns after filtering is done

    df[0] = df[0].str.split('/').str[8:10].apply('_'.join)  # replace file path column with video_name
    df[1] = df[1].str.replace('.jpg', '')  # change filename to file-sequence-number
    df[1] = pd.to_numeric(df[1])
    # df[2] = df[2].str.split('-').str[1]  # From "Celeb-real" to "real"

    grouped_df = df.groupby(0)
    for video_name, filtered_df in tqdm(grouped_df):
        if os.path.exists(os.path.join(output_directory, video_name + '.csv')):  # saved before, skip
            continue

        filtered_df = filtered_df.drop(columns=[0])  # drop folder_name/video_name
        filtered_df = filtered_df.sort_values(1)  # sort by sequence number
        # filtered_df = filtered_df.drop(columns=[1])  # drop sequence number

        filtered_df.to_csv(os.path.join(output_directory, video_name + '.csv'), header=False, index=False)


def make_npy_by_batch_CelebDF(videos_csv_directory, npy_out_dir, which_set='train', method='B', batch_sz=64,
                              timesteps=500):
    class_name_num_map = {'real': 0,
                          'synthesis': 1}

    data_np, i = None, 0
    for video_name_csv in tqdm(os.listdir(videos_csv_directory)):
        filepath = os.path.join(videos_csv_directory, video_name_csv)

        class_name = video_name_csv.split('-')[1].split('_')[0]
        class_num = class_name_num_map[class_name]
        # print(class_name, video_name_csv)

        df = pd.read_csv(filepath, header=None)
        df_splits = split_dataframe(df, chunk_size=timesteps)

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

            if (i + 1) % batch_sz == 0:
                np.save(f'{npy_out_dir}/CelebDF_embeddings_Method_{method}_{which_set}_{(i+1)//batch_sz}.npy', data_np)
                data_np = None
            i += 1

    if data_np is not None:
        np.save(f'{npy_out_dir}/CelebDF_embeddings_Method_{method}_{which_set}_{(i+1)//batch_sz}.npy', data_np)
    # print(data_np.shape)


def main():
    # group_data_by_video_CelebDF(
    #     r'./data/CelebDF_embeddings_Method_B/val_embeddings.csv',
    #     r'./data/CelebDF_embeddings_Method_B_val/',
    #     which_set='test'
    # )
    make_npy_by_batch_CelebDF(
        videos_csv_directory=r'/mnt/sanjay/CelebDF_embeddings_Method_B_train/',
        npy_out_dir=r'/mnt/sanjay/CelebDF_embeddings_Method_B_train_npy',
        which_set='train',
        method='B'
    )


if __name__ == '__main__':
    main()
