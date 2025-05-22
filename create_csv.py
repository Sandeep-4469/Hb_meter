import os
import cv2
import csv
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from tempfile import mkdtemp
from multiprocessing import Pool, cpu_count

input_csv = 'Hb_raw_xlsx.csv'
output_dir = 'processed_features_new'
video_dir = 'Videos'
os.makedirs(output_dir, exist_ok=True)

def random_rotation(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_matrix, (width, height), flags=cv2.INTER_NEAREST)

def process_video(row):
    video_name_raw = row['matched_video']
    video_path = os.path.join(video_dir, video_name_raw)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    final_output_path = os.path.join(output_dir, f"{video_name}_flattened.csv")

    if os.path.exists(final_output_path):
        return video_name, final_output_path

    if not os.path.exists(video_path):
        print(f"[Warning] Video not found: {video_path}")
        return video_name, None

    temp_dir = mkdtemp()
    flattened_data = []

    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"[Error] Cannot open video: {video_path}")
            return video_name, None

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        roi_size = min(frame_width, frame_height) // 3
        roi_top = (frame_height - roi_size) // 2
        roi_left = (frame_width - roi_size) // 2
        roi_bottom = roi_top + roi_size
        roi_right = roi_left + roi_size

        for angle in range(1, 21):
            csv_path = os.path.join(temp_dir, f"{video_name}_angle_{angle}.csv")
            with open(csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Frame", "R", "G", "B", "H", "S", "V", "L", "A", "Grayscale"])
                frame_count = 0
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    if frame_count < 50:
                        frame_count += 1
                        continue
                    if frame_count > 590:
                        break

                    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_rgb_rotated = random_rotation(roi_rgb, angle)
                    roi_hsv = cv2.cvtColor(roi_rgb_rotated, cv2.COLOR_RGB2HSV)
                    roi_lab = cv2.cvtColor(roi_rgb_rotated, cv2.COLOR_RGB2LAB)
                    roi_gray = cv2.cvtColor(roi_rgb_rotated, cv2.COLOR_RGB2GRAY)

                    r, g, b = cv2.split(roi_rgb_rotated)
                    h, s, v = cv2.split(roi_hsv)
                    l, a, _ = cv2.split(roi_lab)

                    for i in range(len(r)):
                        writer.writerow([
                            frame_count,
                            r[i][0], g[i][0], b[i][0],
                            h[i][0], s[i][0], v[i][0],
                            l[i][0], a[i][0], roi_gray[i][0]
                        ])
                    frame_count += 1

            df = pd.read_csv(csv_path)
            channels = ['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'Grayscale']
            histograms = {}

            for channel in channels:
                channel_histograms = []
                for frame in range(50, 590):
                    filtered_df = df[df['Frame'] == frame]
                    hist, _ = np.histogram(filtered_df[channel], bins=256, range=(0, 256), density=True)
                    channel_histograms.append(hist)
                histograms[channel] = np.mean(channel_histograms, axis=0)

            histogram_df = pd.DataFrame(histograms)
            flattened = histogram_df.to_numpy().reshape(1, -1, order='F')
            flattened_data.append(flattened)

        final_data = np.vstack(flattened_data)
        pd.DataFrame(final_data).to_csv(final_output_path, index=False, header=False)
        return video_name, final_output_path

    except Exception as e:
        print(f"[Error] Failed on {video_name}: {e}")
        return video_name, None
    finally:
        video.release()
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    df = pd.read_csv(input_csv)

    # Remove NaN or empty entries in 'matched_video'
    df = df[df['matched_video'].notna()]
    df = df[df['matched_video'].astype(str).str.strip() != '']
    df['matched_video'] = df['matched_video'].astype(str)

    # Only keep rows where the video file exists
    df['video_path'] = df['matched_video'].apply(lambda x: os.path.join(video_dir, x))
    df = df[df['video_path'].apply(os.path.exists)].reset_index(drop=True)

    print(f" {len(df)} valid video(s) found for processing.")

    # Start multiprocessing
    print(" Starting multiprocessing...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_video, df.to_dict(orient='records')), total=len(df)))

    # Merge paths back into DataFrame
    video_to_path = {video: path for video, path in results if path is not None}
    df['csv_path'] = df['matched_video'].map(lambda x: video_to_path.get(os.path.splitext(os.path.basename(x))[0], ''))

    # Save the updated CSV
    final_dataset_path = os.path.join(output_dir, 'updated_dataset.csv')
    df.to_csv(final_dataset_path, index=False)
    print(f"\nDataset with feature paths saved to: {final_dataset_path}")
