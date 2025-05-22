import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import os
import random

class HbVideoDataset(Dataset):
    def __init__(self, csv_file, video_dir='Videos', clip_len=16, frame_size=112, augment=False):
        self.data = pd.read_csv(csv_file)
        self.clip_len = clip_len
        self.frame_size = frame_size
        self.augment = augment
        self.video_dir = video_dir

        # Filter out missing videos
        self.data = self.data[self.data['matched_video'].apply(
            lambda x: os.path.exists(os.path.join(self.video_dir, x))
        )].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def center_roi(self, frame):
        h, w, _ = frame.shape
        roi_size = min(h, w) // 3
        roi_top = (h - roi_size) // 2
        roi_left = (w - roi_size) // 2
        roi_bottom = roi_top + roi_size
        roi_right = roi_left + roi_size
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        return cv2.resize(roi, (self.frame_size, self.frame_size))

    def random_rotation(self, image, max_angle=10):
        angle = random.uniform(-max_angle, max_angle)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)
        return rotated_image

    def load_clip(self, video_path):
        full_path = os.path.join(self.video_dir, video_path)
        cap = cv2.VideoCapture(full_path)
        frames = []
        frame_count = 0
        target_frames = list(range(50, 591))  # inclusive of 590

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count in target_frames:
                roi = self.center_roi(frame)
                if self.augment:
                    roi = self.random_rotation(roi)
                frames.append(roi)
            frame_count += 1
            if frame_count > 590:
                break
        cap.release()

        if len(frames) < self.clip_len:
            raise ValueError(f"Not enough frames extracted from {video_path}")

        # Randomly sample a clip of length clip_len
        idx = np.random.randint(0, len(frames) - self.clip_len + 1)
        clip = np.stack(frames[idx:idx+self.clip_len], axis=0)  # [T, H, W, C]
        clip = clip.transpose(3, 0, 1, 2)  # [C, T, H, W]
        return torch.tensor(clip / 255.0, dtype=torch.float32)

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['matched_video']
        hb_value = self.data.iloc[idx]['hb']
        try:
            clip = self.load_clip(video_path)
        except Exception as e:
            print(f"[Error] Failed to load {video_path}: {e}")
            clip = torch.zeros((3, self.clip_len, self.frame_size, self.frame_size), dtype=torch.float32)
        return clip, torch.tensor(hb_value, dtype=torch.float32)
