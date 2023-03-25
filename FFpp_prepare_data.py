import os

import numpy as np
import pandas as pd
from warnings import simplefilter
from collections import Counter

from tqdm import tqdm

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


def group_data_by_video(input_csv_path, output_directory):
    # df = pd.read_csv(input_csv_path, header=None)
    df_chunks = pd.read_csv(input_csv_path, index_col=False, chunksize=10000)
    df = pd.concat(df_chunks)

    print('\nDATA READING DONE.\n\n')
    # df = df.drop(columns=[0, 3])
    df = df.drop(columns=['image_path', 'run_type', 'target', 'logit_1', 'logit_2'])
    # grouped_df = df.groupby(2)
    grouped_df = df.groupby('folder')  # folder == video_name

    for video_name, filtered_df in tqdm(grouped_df):
        if os.path.exists(os.path.join(output_directory, video_name+'.csv')):
            continue

        # print(video_name)
        # filtered_df_process = filtered_df_process.drop(columns=[2])
        filtered_df_process = filtered_df.copy()
        filtered_df_process['filename'] = filtered_df_process['filename'].str.replace('.jpg', '')
        try:
            filtered_df_process['filename'] = pd.to_numeric(filtered_df_process['filename'])
        except ValueError:
            continue
        # print(filtered_df_process.head())

        filtered_df_process = filtered_df_process.sort_values('filename')
        filtered_df_process = filtered_df_process.drop(columns=['folder'])

        filtered_df_process = filtered_df_process.drop_duplicates(subset=['filename'])
        # print(filtered_df_process.head())
        # exit(1)

        filtered_df_process.to_csv(os.path.join(output_directory, video_name+'.csv'), header=False, index=False)


def main():
    # df_chunks = pd.read_csv(r'./data/FFpp_test_embeddings_cleaned.csv', chunksize=10000, low_memory=False, index_col=False)
    # df = pd.concat(df_chunks)
    # df.head(2000).to_csv(r'./data/FFpp_test_embeddings_short.csv', index=False)
    # group_data_by_video(r'./data/FFpp_test_embeddings_cleaned.csv',
    #                     r'./data/FFpp_test_embeddings/')

    videos_csv_path = r'data/FFpp_embeddings_Method_A_train/'
    class_name_num_map = {'original': 0,
                          'Face2Face': 1,
                          'NeuralTextures': 2,
                          'Deepfakes': 3,
                          'FaceSwap': 4,
                          'FaceShifter': 5}

    data_np = None
    for i, video_name_csv in enumerate(tqdm(os.listdir(videos_csv_path))):
        filepath = os.path.join(videos_csv_path, video_name_csv)

        class_name = video_name_csv.split('_')[0]
        class_num = class_name_num_map[class_name]
        # print(class_name, video_name_csv)

        df = pd.read_csv(filepath, header=None)

        df = df.drop(columns=[0])  # remove sequence number
        df.insert(0, 0, int(class_num))  # set class number
        # print(df.head())

        df_np = df.to_numpy()
        df_np = np.expand_dims(df_np, axis=0)  # 1 x n_timesteps x embedding_length

        # padding to match n_timesteps for all videos, set to 500
        df_np = np.pad(df_np, ((0, 0), (0, 500-df_np.shape[1]), (0, 0)), 'constant', constant_values=0)

        if data_np is None:
            data_np = df_np
        else:
            data_np = np.vstack([data_np, df_np])

        if (i+1) % 64 == 0:
            np.save(f'data/FFpp_embeddings_Method_A/FFpp_embeddings_Method_A_train_{(i+1)//64}.npy', data_np)
            data_np = None
        # print(df_np.shape, data_np.shape)
        # print(df_np[:10])
        # exit(0)
    np.save('data/FFpp_embeddings_Method_A/FFpp_embeddings_Method_A_train_final.npy', data_np)
    print(data_np.shape)


if __name__ == '__main__':
    # check_np_padding()
    main()
