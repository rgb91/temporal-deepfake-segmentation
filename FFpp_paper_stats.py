import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from utils import remove_extension


def fake_ratio_one_segment():
    data_path = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings/FFpp_temporal_eval_data/one_segment'
    gt_file = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings/FFpp_temporal_eval_data' \
              r'/temporal_one_segment_ground_truth.csv'
    gt_df = pd.read_csv(gt_file, index_col=False)

    datasets = ['fake_DF', 'fake_FSh', 'fake_F2F', 'fake_NT', 'fake_FS']

    total_fake_length, total_video_length = 0, 0
    for d in datasets:
        d_path = os.path.join(data_path, d)
        d_fake_length, d_video_length = 0, 0

        for v_name in os.listdir(d_path):
            v_preds_df = pd.read_csv(os.path.join(d_path, v_name), index_col=False)

            gt_data = gt_df[gt_df['video_name'] == remove_extension(v_name)].iloc[0]
            video_length = max(gt_data['total_frames'], len(v_preds_df))

            fake_start = gt_data['fake_start']
            fake_end = gt_data['fake_end']
            fake_length = fake_end - fake_start

            d_fake_length += fake_length
            d_video_length += video_length

        d_ratio = d_fake_length / d_video_length
        total_fake_length += d_fake_length
        total_video_length += d_video_length
        print(f'Dataset: {d}\nRatio: {d_ratio:.04f}\n')

    print(f'Total ratio: {total_fake_length / total_video_length:.04f}\n')


def fake_ratio_two_segments():
    data_path = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings/FFpp_temporal_eval_data/two_segments'
    gt_file = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings/FFpp_temporal_eval_data' \
              r'/temporal_two_segments_ground_truth.csv'
    gt_df = pd.read_csv(gt_file, index_col=False)

    datasets = ['fake_DF', 'fake_FSh', 'fake_F2F', 'fake_NT', 'fake_FS']

    total_fake_length, total_video_length = 0, 0
    for d in datasets:
        d_path = os.path.join(data_path, d)
        d_fake_length, d_video_length = 0, 0

        for v_name in os.listdir(d_path):
            v_preds_df = pd.read_csv(os.path.join(d_path, v_name), index_col=False)

            gt_data = gt_df[gt_df['video_name'] == remove_extension(v_name)].iloc[0]
            video_length = max(gt_data['total_frames'], len(v_preds_df))

            fake1_start = gt_data['fake1_start']
            fake1_end = gt_data['fake1_end']
            fake2_start = gt_data['fake2_start']
            fake2_end = gt_data['fake2_end']
            fake_length = (fake1_end - fake1_start) + (fake2_end - fake2_start)

            d_fake_length += fake_length
            d_video_length += video_length

        d_ratio = d_fake_length / d_video_length
        total_fake_length += d_fake_length
        total_video_length += d_video_length
        print(f'Dataset: {d}\nRatio: {d_ratio:.04f}\n')

    print(f'Total ratio: {total_fake_length / total_video_length:.04f}\n')


if __name__ == '__main__':
    fake_ratio_two_segments()