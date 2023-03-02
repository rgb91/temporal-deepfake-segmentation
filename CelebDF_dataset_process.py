"""
CelebDF Download links
Google Drive:
Celeb-DF (v2) dataset: 
https://drive.google.com/open?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj. 

Celeb-DF (v1) dataset: 
https://drive.google.com/open?id=10NGF38RgF8FZneKOuCOdRIsPzpC7_WDd. 

Baidu Net Disk:
Celeb-DF (v2) dataset: 
https://pan.baidu.com/s/1EcYX0s4U3kbI1V2vdrP46A 
code:yxa1 

Celeb-DF (v1) dataset: 
https://pan.baidu.com/s/16QulfMFG4TQB9iMZIZnsjQ 
code:ku0s 

(Refresh if the exception is occurred)

Converts video_to_frames_all_folders including align_and_crop_face_from_frames
"""
import os
import cv2
import shutil
import threading
from tqdm import tqdm
from pathlib import Path
from deepface import DeepFace


KPS = 0  # if 0 or negative KPS=FPS i.e. skip NO frames
KPS_string = str(KPS) if KPS > 0 else 'all'
IMAGE_EXTENSION = '.jpg'  # .png takes way more space
FACE_RESOLUTION = 256
VIDEO_HOME = r'/data/Datasets/Celeb-DF-v2'
FRAMES_HOME = rf'/data/Datasets/Celeb-DF-v2-frames'
ALIGNED_FRAMES_HOME = rf'/data/Datasets/Celeb-DF-v2-frames-aligned-{FACE_RESOLUTION}-KPS_{KPS_string}'
SPLIT_FRAMES_HOME = rf'/data/Datasets/Celeb-DF-v2-frames-aligned-{FACE_RESOLUTION}-KPS_{KPS_string}-split'
CLASSES = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']


def video_to_frames_one_folder(source_dir, destination_dir, aligned_dest_dir, overwrite=False):
    print('START:', destination_dir)
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    Path(aligned_dest_dir).mkdir(parents=True, exist_ok=True)

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
        KPS_divider = fps if KPS <= 0 else KPS
        count, hop = 0, round(fps//KPS_divider)  # hop = round(fps/kps) where kps is keyframes per sec.

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
        source_dir = os.path.join(VIDEO_HOME, c)
        destination_dir = os.path.join(FRAMES_HOME, c)
        aligned_dest_dir = os.path.join(ALIGNED_FRAMES_HOME, c)

        t = threading.Thread(target=video_to_frames_one_folder, args=(source_dir, destination_dir, aligned_dest_dir, overwrite))
        threads.append(t)
        t.start()
    for _t in threads:
        _t.join()
    
    print('\n==============   END   =================\n')
    return None


def train_test_split():
    src_home, dest_home = ALIGNED_FRAMES_HOME, SPLIT_FRAMES_HOME
    for i, cls in enumerate(CLASSES):
        vids = os.listdir(os.path.join(src_home, cls))
        vids, split_loc = sorted(vids), int(len(vids)*0.8)
        train_split, test_split = vids[:split_loc], vids[split_loc:]
        print(f'{cls}: {len(train_split)}, {len(test_split)}')
        train_dir = os.path.join(dest_home, 'train', cls)
        test_dir = os.path.join(dest_home, 'test', cls)
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        for vid in tqdm(train_split, desc='Train'):
            if vid=='train' or vid=='test': continue
            source, destination = os.path.join(src_home, cls, vid), os.path.join(train_dir, vid)
            shutil.copytree(source, destination)
        for vid in tqdm(test_split, desc='Test'):
            if vid=='train' or vid=='test': continue
            source, destination = os.path.join(src_home, cls, vid), os.path.join(test_dir, vid)
            shutil.copytree(source, destination)
    return None


if __name__ == '__main__':
    # video_to_frames_all_folders(overwrite=True)
    # align_and_crop_face_from_frames()
    train_test_split()