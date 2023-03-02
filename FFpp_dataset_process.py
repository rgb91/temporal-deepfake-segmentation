"""
Converts video_to_frames_all_folders including align_and_crop_face_from_frames
"""
import os
import cv2
import shutil
import threading
from tqdm import tqdm
from pathlib import Path
from deepface import DeepFace


KPS = 5
COMPRESSION = 'c23'  # 'raw' takes up the whole HDD
IMAGE_EXTENSION = '.jpg'  # .png takes way more space
FACE_RESOLUTION = 224
VIDEO_HOME = r'/data/Datasets/FaceForensicsPP'
FRAMES_HOME = rf'/data/Datasets/FFpp-{COMPRESSION}-frames'
ALIGNED_FRAMES_HOME = rf'/data/Datasets/FFpp-{COMPRESSION}-frames-aligned-{FACE_RESOLUTION}'
SPLIT_FRAMES_HOME = rf'/data/Datasets/FFpp-{COMPRESSION}-frames-aligned-{FACE_RESOLUTION}-split'
CLASSES = ['manipulated_sequences', 'original_sequences']
DATASETS = [['DeepFakeDetection', 'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures'], ['actors', 'youtube']]


def video_to_frames_one_folder(source_dir, destination_dir, aligned_dest_dir, overwrite=False):
    print('START:', destination_dir)
    for video_filename in tqdm(os.listdir(source_dir)):
        if not video_filename.endswith('.mp4'): continue
        
        video_name = video_filename[:-4]
        video_filepath = os.path.join(source_dir, video_filename)
        frames_dir = os.path.join(destination_dir, video_name)
        aligned_face_dir = os.path.join(aligned_dest_dir, video_name)
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        Path(aligned_face_dir).mkdir(parents=True, exist_ok=True)

        vidcap = cv2.VideoCapture(video_filepath)
        success, image = vidcap.read()
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        count, hop = 0, round(fps//KPS)  # hop = round(fps/kps) where kps is keyframes per sec.

        while success:
            if count % hop == 0:
                # Extract frame
                frame_name = f'{count:05d}{IMAGE_EXTENSION}'
                frame_filepath = os.path.join(frames_dir, frame_name)
                if overwrite:
                    cv2.imwrite(frame_filepath, image)
                else:
                    if not os.path.exists(frame_filepath):
                        cv2.imwrite(frame_filepath, image)
                
                # Detect, Align, Crop, save face
                aligned_face_image_path = os.path.join(aligned_face_dir, frame_name)
                try:
                    aligned_face_image = DeepFace.detectFace(img_path=frame_filepath, 
                                                            target_size=(FACE_RESOLUTION, FACE_RESOLUTION), 
                                                            detector_backend='retinaface',
                                                            enforce_detection=True)
                    cv2.imwrite(aligned_face_image_path, aligned_face_image[:, :, ::-1]*255)
                except ValueError:
                    err_image_dir = os.path.join(aligned_face_dir, 'face_not_found')
                    Path(err_image_dir).mkdir(parents=True, exist_ok=True)
                    err_image_path = os.path.join(err_image_dir, frame_name)
                    shutil.copyfile(frame_filepath, err_image_path)
            
            count += 1
            success, image = vidcap.read()
        vidcap.release()


def video_to_frames_all_folders(overwrite=False):
    threads = []
    for i, c in enumerate(CLASSES):
        for d in DATASETS[i]:
            source_dir = os.path.join(VIDEO_HOME, c, d, COMPRESSION, 'videos')
            destination_dir = os.path.join(FRAMES_HOME, c, d)
            aligned_dest_dir = os.path.join(ALIGNED_FRAMES_HOME, c, d)
            Path(destination_dir).mkdir(parents=True, exist_ok=True)

            t = threading.Thread(target=video_to_frames_one_folder, args=(source_dir, destination_dir, aligned_dest_dir, overwrite))
            threads.append(t)
            t.start()
    for _t in threads:
        _t.join()
    
    print('\n==============   END   =================\n')
    return None


def align_and_crop_worker(source_dir, destination_dir, detector_backend):
    print('START:', destination_dir)
    face_not_found_count = 0
    for frame_name in os.listdir(source_dir):
        src_image_path = os.path.join(source_dir, frame_name)
        if os.path.isdir(src_image_path): continue
        dst_image_path = os.path.join(destination_dir, frame_name)
        try:
            aligned_face_image = DeepFace.detectFace(img_path=src_image_path, 
                                                    target_size=(224, 224), 
                                                    detector_backend=detector_backend,
                                                    enforce_detection=True)
            cv2.imwrite(dst_image_path, aligned_face_image)
        except ValueError:
            err_image_dir = os.path.join(source_dir, 'face_not_found')
            if not os.path.exists(err_image_dir): os.mkdir(err_image_dir)
            # err_image_path = os.path.join(err_image_dir, frame_name)
            shutil.copyfile(src_image_path, err_image_dir)
            face_not_found_count += 1
        if aligned_face_image: cv2.imwrite(dst_image_path, aligned_face_image)
    print('Total face not found:', face_not_found_count, '\nDestination:', destination_dir)


def align_and_crop_face_from_frames():
    threads = []
    for i, cls in enumerate(CLASSES):
        for dataset in DATASETS[i]:
            for vid_name in os.listdir(os.path.join(FRAMES_HOME, cls, dataset)):
                
                source_dir = os.path.join(FRAMES_HOME, cls, dataset, vid_name)
                destination_dir = os.path.join(FRAMES_HOME, cls, dataset, vid_name, 'aligned')
                Path(destination_dir).mkdir(parents=True, exist_ok=True)
                
                t = threading.Thread(target=align_and_crop_worker, args=(source_dir, destination_dir, 'mediapipe'))
                threads.append(t)
                t.start()
    
    for _t in threads:
        _t.join()
    
    print('\n==============   END   =================\n')
    return None


def train_test_split():
    home = ALIGNED_FRAMES_HOME
    for i, cls in enumerate(CLASSES):
        for dataset in DATASETS[i]:
            vids = os.listdir(os.path.join(home, cls, dataset))
            vids, split_loc = sorted(vids), int(len(vids)*0.8)
            train_split, test_split = vids[:split_loc], vids[split_loc:]
            print(f'{cls}/{dataset}: {len(train_split)}, {len(test_split)}')
            if not os.path.exists(os.path.join(home, cls, dataset, 'train')): 
                os.mkdir(os.path.join(home, cls, dataset, 'train'))
            if not os.path.exists(os.path.join(home, cls, dataset, 'test')): 
                os.mkdir(os.path.join(home, cls, dataset, 'test'))
            for vid in tqdm(train_split, desc='Train'):
                if vid=='train' or vid=='test': continue
                source, destination = os.path.join(home, cls, dataset, vid), os.path.join(home, cls, dataset, 'train', vid)
                # print(source, '\n', destination, '\n')
                shutil.move(source, destination)
            for vid in tqdm(test_split, desc='Test'):
                if vid=='train' or vid=='test': continue
                source, destination = os.path.join(home, cls, dataset, vid), os.path.join(home, cls, dataset, 'test', vid)
                # print(source, '\n', destination, '\n')
                shutil.move(source, destination)
    return None


def train_test_split_v2():
    src_home, dest_home = ALIGNED_FRAMES_HOME, SPLIT_FRAMES_HOME
    for i, cls in enumerate(CLASSES):
        for dataset in DATASETS[i]:
            vids = os.listdir(os.path.join(src_home, cls, dataset))
            vids, split_loc = sorted(vids), int(len(vids)*0.8)
            train_split, test_split = vids[:split_loc], vids[split_loc:]
            print(f'{cls}/{dataset}: {len(train_split)}, {len(test_split)}')
            train_dir = os.path.join(dest_home, 'train', cls, dataset)
            test_dir = os.path.join(dest_home, 'test', cls, dataset)
            Path(train_dir).mkdir(parents=True, exist_ok=True)
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            for vid in tqdm(train_split, desc='Train'):
                if vid=='train' or vid=='test': continue
                source, destination = os.path.join(src_home, cls, dataset, vid), os.path.join(train_dir, vid)
                shutil.copytree(source, destination)
            for vid in tqdm(test_split, desc='Test'):
                if vid=='train' or vid=='test': continue
                source, destination = os.path.join(src_home, cls, dataset, vid), os.path.join(test_dir, vid)
                shutil.copytree(source, destination)
    return None


if __name__ == '__main__':
    # video_to_frames_all_folders(overwrite=True)
    # align_and_crop_face_from_frames()
    train_test_split_v2()