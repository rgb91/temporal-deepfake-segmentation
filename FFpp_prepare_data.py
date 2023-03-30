import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from warnings import simplefilter

from utils import make_short_file, split_dataframe

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO, filename='./FFpp_temporal_evaluation_errors.log')


def get_class_name_from_video_name(video_name):
    v_parts = video_name.split('_')
    if v_parts[0] == 'fake':
        v_name = '_'.join([v_parts[0], v_parts[1]])
    else:
        v_name = v_parts[0]
    return v_name


def verify_train_embeddings(dir):
    """
    Output

    fake_F2F: 800
    fake_NT: 35
    fake_DF: 800
    real: 800
    fake_FS: 800
    """
    # path = os.path.join(DATA_ROOT_FFpp, 'FFpp_train_embeddings_clean.csv')
    # df_chunks = pd.read_csv(path, index_col=False, chunksize=10000)
    # df = pd.concat(df_chunks)
    # df_fake_NT = df.loc[df['folder'].str.startswith('fake_NT')]
    # grouped_df = df_fake_NT.groupby('folder')  # folder == video_name
    # print(len(grouped_df))

    count_map = dict()
    for video_name in os.listdir(dir):
        v_name = get_class_name_from_video_name(video_name)
        if v_name not in count_map:
            count_map[v_name] = 1
        else:
            count_map[v_name] += 1

    for v, c in count_map.items():
        print(f'{v}: {c}')
    print('\n')


def clean_raw_csv_FFpp_main(infile, outfile):
    import csv
    columns = ['image_path', 'filename', 'folder', 'run_type', 'target', 'logit_1', 'logit_2']
    for i in range(768):
        columns.append(f'e{i:03d}')

    first_row = True
    with open(infile, encoding='utf-8') as f, open(outfile, 'w') as o:
        reader = csv.reader(f)
        writer = csv.writer(o, delimiter=',')  # adjust as necessary
        for r_row in reader:
            if first_row:
                first_row = False
                w_row = columns
            else:
                w_row = [item for item in r_row]
            writer.writerow(w_row)
    print('Done')


def group_data_by_video_FFpp(infile, out_dir):
    df_chunks = pd.read_csv(infile, index_col=False, chunksize=10000)
    df = pd.concat(df_chunks)

    print('\nDATA READING DONE.\n\n')
    df = df.drop(columns=['image_path', 'run_type', 'target', 'logit_1', 'logit_2'])
    grouped_df = df.groupby('folder')  # folder == video_name

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for video_name, filtered_df in tqdm(grouped_df):
        if os.path.exists(os.path.join(out_dir, video_name + '.csv')):
            continue

        filtered_df_process = filtered_df.copy()
        filtered_df_process['filename'] = filtered_df_process['filename'].str.replace('.jpg', '')
        try:
            filtered_df_process['filename'] = pd.to_numeric(filtered_df_process['filename'])
        except ValueError:
            continue

        filtered_df_process = filtered_df_process.sort_values('filename')
        filtered_df_process = filtered_df_process.drop(columns=['folder'])

        filtered_df_process = filtered_df_process.drop_duplicates(subset=['filename'])
        # print(filtered_df_process.head())
        # exit(1)

        filtered_df_process.to_csv(os.path.join(out_dir, video_name + '.csv'), header=False, index=False)


def group_data_by_video_FFpp_temporal(infile, out_dir):
    real, fake = 0, 1

    df_chunks = pd.read_csv(infile, index_col=False, chunksize=10000)
    df = pd.concat(df_chunks)
    df = df.drop(columns=['target', 'logit_1', 'logit_2'])
    print('\nDATA READING DONE.\n\n')

    df['image_path'] = df['image_path'].str.replace(f'/data/gpfs/projects/punim1875/FFpp_temporal_one_segment/', '')
    df['dataset'] = df['image_path'].str.split('/').str[0]
    df['video_name'] = df['dataset'] + '_' + df['folder']

    gt_file = r'./FFpp_temporal_eval_data/temporal_one_segment_ground_truth.csv'
    gt_df = pd.read_csv(gt_file, index_col=False)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    grouped_df = df.groupby('video_name')  # folder == video_name
    for video_name, filtered_df in tqdm(grouped_df):
        if os.path.exists(os.path.join(out_dir, video_name + '.csv')):
            continue
        filtered_df_process = filtered_df.copy()
        filtered_df_process['filename'] = filtered_df_process['filename'].str.replace('.jpg', '')
        filtered_df_process['filename'] = pd.to_numeric(filtered_df_process['filename'])
        filtered_df_process = filtered_df_process.sort_values('filename')

        v_name = video_name.split('_')[2] + '_' + video_name.split('_')[3]
        gt_data = gt_df[gt_df['video_name'] == v_name].iloc[0]
        total_frames = max(gt_data['total_frames'], len(filtered_df))
        fake_start = gt_data['fake_start']
        fake_end = gt_data['fake_end']
        gt_labels = [real] * (fake_start) + [fake] * (fake_end - fake_start) + [real] * (total_frames - fake_end)

        filtered_df_process = filtered_df_process.drop(columns=['image_path', 'folder', 'video_name', 'dataset'])
        filtered_df_process.insert(1, 'target', gt_labels)

        # filtered_df_process = filtered_df_process.drop_duplicates(subset=['filename'])
        # print(filtered_df_process.head())
        # exit(1)

        filtered_df_process.to_csv(os.path.join(out_dir, video_name + '.csv'), header=False, index=False)


def make_npy_batch_FFpp_old():
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
        df_np = np.pad(df_np, ((0, 0), (0, 500 - df_np.shape[1]), (0, 0)), 'constant', constant_values=0)

        if data_np is None:
            data_np = df_np
        else:
            data_np = np.vstack([data_np, df_np])

        if (i + 1) % 64 == 0:
            np.save(f'data/FFpp_embeddings_Method_A/FFpp_embeddings_Method_A_train_{(i + 1) // 64}.npy', data_np)
            data_np = None
        # print(df_np.shape, data_np.shape)
        # print(df_np[:10])
        # exit(0)
    np.save('data/FFpp_embeddings_Method_A/FFpp_embeddings_Method_A_train_final.npy', data_np)
    print(data_np.shape)


def make_npy_by_batch_FFpp(in_dir, out_dir, which_set, batch_sz=64, timesteps=500, binary=False, overlap=0):
    if binary:
        class_name_num_map = {'real': 0, 'fake_F2F': 1, 'fake_NT': 1, 'fake_DF': 1, 'fake_FS': 1, 'fake_FSh': 1}
    else:
        class_name_num_map = {'real': 0, 'fake_F2F': 1, 'fake_NT': 2, 'fake_DF': 3, 'fake_FS': 4, 'fake_FSh': 5}

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data_np, i = None, 0
    for video_name_csv in tqdm(os.listdir(in_dir)):
        class_name = get_class_name_from_video_name(video_name_csv)
        class_num = class_name_num_map[class_name]

        filepath = os.path.join(in_dir, video_name_csv)
        df = pd.read_csv(filepath, header=None)
        if overlap > 0:
            df_splits = [df.loc[i:i+timesteps-1, :] for i in range(0, len(df), timesteps-overlap) if i < len(df)-overlap]
        else:
            df_splits = split_dataframe(df, chunk_size=timesteps)

        for df_split in df_splits:
            df_split = df_split.drop(columns=[0])  # remove sequence number
            df_split.insert(0, 'class', int(class_num))  # set class number
            df_np = df_split.to_numpy()
            df_np = np.expand_dims(df_np, axis=0)  # 1 x n_timesteps x embedding_length

            if overlap < (timesteps-1):  # if overlap == timesteps-1 then now need to add padding
                df_np = np.pad(df_np, ((0, 0), (0, timesteps - df_np.shape[1]), (0, 0)), 'constant', constant_values=0)

            if data_np is None:
                data_np = df_np
            else:
                data_np = np.vstack([data_np, df_np])

            if (i + 1) % batch_sz == 0:
                np.save(f'{out_dir}/FFpp_{which_set}_embeddings_{(i + 1) // batch_sz}.npy',
                        data_np)
                data_np = None
            i += 1

    if data_np is not None:
        np.save(f'{out_dir}/FFpp_{which_set}_embeddings_{(i + 1) // batch_sz}.npy', data_np)


def make_npy_by_video_FFpp(in_dir, out_dir, timesteps=500, binary=False, overlap=0):
    if binary:
        class_name_num_map = {'real': 0, 'fake_F2F': 1, 'fake_NT': 1, 'fake_DF': 1, 'fake_FS': 1, 'fake_FSh': 1}
    else:
        class_name_num_map = {'real': 0, 'fake_F2F': 1, 'fake_NT': 2, 'fake_DF': 3, 'fake_FS': 4, 'fake_FSh': 5}

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for video_name_csv in tqdm(os.listdir(in_dir)):
        filepath = os.path.join(in_dir, video_name_csv)

        class_name = get_class_name_from_video_name(video_name_csv)
        class_num = class_name_num_map[class_name]

        df = pd.read_csv(filepath, header=None)
        if overlap > 0:
            df_splits = [df.loc[i:i + timesteps - 1, :] for i in range(0, len(df), timesteps - overlap) if
                         i < len(df) - overlap]
        else:
            df_splits = split_dataframe(df, chunk_size=timesteps)

        data_np = None
        for df_split in df_splits:
            df_split = df_split.drop(columns=[0])  # remove sequence number
            df_split.insert(0, 'class', int(class_num))  # set class number
            df_np = df_split.to_numpy()
            df_np = np.expand_dims(df_np, axis=0)  # 1 x n_timesteps x embedding_length

            if overlap < (timesteps-1):  # if overlap == timesteps-1 then now need to add padding
                df_np = np.pad(df_np, ((0, 0), (0, timesteps - df_np.shape[1]), (0, 0)), 'constant', constant_values=0)

            if data_np is None:
                data_np = df_np
            else:
                data_np = np.vstack([data_np, df_np])

        video_name = video_name_csv.split('.')[0]
        np.save(f'{out_dir}/{video_name}.npy', data_np)


def make_npy_for_temporal_data(in_dir, out_dir, timesteps=25, overlap=0):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for video_name_csv in tqdm(os.listdir(in_dir)):
        filepath = os.path.join(in_dir, video_name_csv)

        df = pd.read_csv(filepath, header=None)
        if overlap > 0:
            df_splits = [df.loc[i:i + timesteps - 1, :] for i in range(0, len(df), timesteps - overlap) if
                         i < len(df) - overlap]
        else:
            df_splits = split_dataframe(df, chunk_size=timesteps)
        data_np = None
        for df_split in df_splits:
            df_split = df_split.drop(columns=[0])  # remove sequence number

            df_np = df_split.to_numpy()
            df_np = np.expand_dims(df_np, axis=0)  # 1 x n_timesteps x embedding_length

            if overlap < (timesteps-1):  # if overlap == timesteps-1 then now need to add padding
                df_np = np.pad(df_np, ((0, 0), (0, timesteps - df_np.shape[1]), (0, 0)), 'constant', constant_values=0)

            if data_np is None:
                data_np = df_np
            else:
                data_np = np.vstack([data_np, df_np])

        video_name = video_name_csv.split('.')[0]
        np.save(f'{out_dir}/{video_name}.npy', data_np)


if __name__ == '__main__':
    """
    Four step process:
        1. make_short_file: since the csv files are too big to load to view using editors we make a shorter version
            first to manually verify the file.
        2. clean_raw_csv_FFpp_main: fix any improper things in the csv e.g. wrong headers, duplicate data.
        3. group_data_by_video_FFpp: create individual CSV files for each video.
        4. make_npy_batch_FFpp: create npy files one for each batch, makes life easy in writing a pytorch dataloader.
    """
    DATA_ROOT = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings'

    # make_short_file(infile=os.path.join(DATA_ROOT_FFpp, 'fake_FSh_embeddings.csv'),
    #                 outfile=os.path.join(DATA_ROOT_FFpp, 'fake_FSh_embeddings_short.csv'),
    #                 use_pandas=False,
    #                 sample_size=1000)

    # clean_raw_csv_FFpp_main(infile=os.path.join(DATA_ROOT_FFpp, 'fake_FSh_embeddings.csv'),
    #                         outfile=os.path.join(DATA_ROOT_FFpp, 'fake_FSh_embeddings_clean.csv'))

    # group_data_by_video_FFpp(infile=os.path.join(DATA_ROOT_FFpp, 'fake_FSh_embeddings_clean.csv'),
    #                          out_dir=os.path.join(DATA_ROOT_FFpp, 'fake_FSh_train_embeddings_csv'))

    # verify_train_embeddings(os.path.join(DATA_ROOT_FFpp, 'FFpp_train_embeddings_csv'))

    # make_npy_by_batch_FFpp(in_dir=os.path.join(DATA_ROOT, 'FFpp_train_embeddings_csv'),
    #                        out_dir=os.path.join(DATA_ROOT, 'FFpp_train_embeddings_npy_batches_10steps_binary_overlap'),
    #                        which_set='train',
    #                        timesteps=10,
    #                        binary=True,
    #                        overlap=9)

    make_npy_by_batch_FFpp(in_dir=os.path.join(DATA_ROOT, 'FFpp_test_embeddings_csv'),
                           out_dir=os.path.join(DATA_ROOT, 'FFpp_test_embeddings_npy_batches_10steps_binary_overlap'),
                           which_set='test',
                           timesteps=10,
                           binary=True,
                           overlap=9)

    make_npy_by_video_FFpp(in_dir=os.path.join(DATA_ROOT, 'FFpp_test_embeddings_csv'),
                           out_dir=os.path.join(DATA_ROOT, 'FFpp_test_embeddings_npy_videos_10steps_binary_overlap'),
                           timesteps=10,
                           binary=True,
                           overlap=9)

    # group_data_by_video_FFpp_temporal(infile=os.path.join(DATA_ROOT_FFpp,
    # 'FFpp_temporal_one_segment_embeddings_clean.csv'), out_dir=os.path.join(DATA_ROOT_FFpp,
    # 'FFpp_temporal_one_segment_csv'))

    make_npy_for_temporal_data(in_dir=os.path.join(DATA_ROOT, 'FFpp_temporal_one_segment_csv'),
                               out_dir=os.path.join(DATA_ROOT, 'FFpp_temporal_one_segment_npy_10steps_overlap'),
                               timesteps=10,
                               overlap=9)

    # Drafting Binary classification changes for FFpp from utils import load_data_single_npy, load_data_multiple_npy
    # x, y = load_data_single_npy(r'/data/PROJECT
    # FILES/DFD_Embeddings/FFpp_embeddings/FFpp_test_embeddings_npy_batches_25steps/FFpp_test_embeddings_1.npy') x,
    # y = load_data_multiple_npy(r'/data/PROJECT
    # FILES/DFD_Embeddings/FFpp_embeddings/FFpp_test_embeddings_npy_batches_25steps', 'FFpp_test_embeddings') print(
    # y[:30]) print(x.shape, y.shape) print(Counter(y)) y = np.array([1 if val >= 1 else 0 for val in list(y)])
    # print(x.shape, y.shape) print(y[:30])