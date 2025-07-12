#!/home/nishad/Nishad_env/bin/python

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import cv2
import csv
import itertools
import time
import subprocess
import threading
import RPi.GPIO as GPIO
from rpi_lcd import LCD
import torch.nn.functional as F
from signal import signal, SIGTERM, SIGHUP
import logging

# Setup logging
logging.basicConfig(filename='nishad.log', level=logging.ERROR)

# Change working directory
os.chdir("/home/hbmeter/Hb_meter/")

def unique_file(basename, ext="mkv"):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    name = basename.split("/")[-1]
    k = ""
    while os.path.exists(actualname):
        k = next(c)
        actualname = "%s%d.%s" % (basename, k, ext)
    name = name + f"{k}"
    return name, actualname

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += self.shortcut(residual)
        return F.relu(out)

class DeepHbNetFlexible(nn.Module):
    def __init__(self, input_size=2304, dropout_rate=0.2, num_blocks=4):
        super(DeepHbNetFlexible, self).__init__()
        self.red_branch = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.orange_branch = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.yellow_branch = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        combined_size = 1024 * 3
        hidden_sizes = [1024, 512, 256, 128]
        if num_blocks < 1 or num_blocks > len(hidden_sizes):
            raise ValueError(f"num_blocks must be between 1 and {len(hidden_sizes)}")
        self.blocks = nn.ModuleList()
        in_features = combined_size
        for i in range(num_blocks):
            out_features = hidden_sizes[i]
            self.blocks.append(ResidualBlock(in_features, out_features, dropout_rate))
            in_features = out_features
        self.output = nn.Linear(in_features, 1)

    def forward(self, x_red, x_orange, x_yellow):
        red_out = self.red_branch(x_red)
        orange_out = self.orange_branch(x_orange)
        yellow_out = self.yellow_branch(x_yellow)
        x = torch.cat((red_out, orange_out, yellow_out), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.output(x)

def check_device(device="/dev/video0"):
    if not os.path.exists(device):
        print(f"Error: Video device {device} not found.")
        logging.error(f"Video device {device} not found.")
        exit(1)
    if not os.access(device, os.R_OK | os.W_OK):
        print(f"Error: No read/write permission for {device}.")
        logging.error(f"No read/write permission for {device}.")
        exit(1)

def start_dual(output_file, timer=2, fps=30):
    def led_light():
        RED_PIN = 26
        GREEN_PIN = 19
        BLUE_PIN = 13
        FREQ = 100
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(RED_PIN, GPIO.OUT)
            GPIO.setup(GREEN_PIN, GPIO.OUT)
            GPIO.setup(BLUE_PIN, GPIO.OUT)
            red_pwm = GPIO.PWM(RED_PIN, FREQ)
            green_pwm = GPIO.PWM(GREEN_PIN, FREQ)
            blue_pwm = GPIO.PWM(BLUE_PIN, FREQ)
            red_pwm.start(0)
            green_pwm.start(0)
            blue_pwm.start(0)

            def set_color(red_pwm, green_pwm, blue_pwm, r, g, b):
                red_pwm.ChangeDutyCycle(r)
                green_pwm.ChangeDutyCycle(g)
                blue_pwm.ChangeDutyCycle(b)

            print("Turning on Red for 2 seconds")
            set_color(red_pwm, green_pwm, blue_pwm, 100, 0, 0)
            time.sleep(timer)
            print("Turning on Orange for 2 seconds")
            set_color(red_pwm, green_pwm, blue_pwm, 100, 25, 0)
            time.sleep(timer)
            print("Turning on Yellow for 2 seconds")
            set_color(red_pwm, green_pwm, blue_pwm, 100, 65, 0)
            time.sleep(timer)
        except Exception as e:
            print(f"GPIO error: {e}")
            logging.error(f"GPIO error: {e}")
        finally:
            try:
                red_pwm.stop()
                green_pwm.stop()
                blue_pwm.stop()
                GPIO.cleanup()
            except Exception as e:
                print(f"Error cleaning up GPIO: {e}")
                logging.error(f"Error cleaning up GPIO: {e}")

    def record_video():
        command = [
            "ffmpeg",
            "-y",
            "-f", "v4l2",
            "-video_size", "1920x1080",
            "-input_format", "mjpeg",
            "-i", "/dev/video0",
            "-c:v", "copy",
            "-t", str(timer * 3),
            "-r", str(fps),
            output_file
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            print("Ok:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            logging.error(f"FFmpeg error: {e.stderr}")
        except Exception as e:
            print(f"Exception in record_video: {e}")
            logging.error(f"Exception in record_video: {e}")

    check_device()
    led_thread = threading.Thread(target=led_light)
    video_thread = threading.Thread(target=record_video)
    led_thread.start()
    video_thread.start()
    led_thread.join()
    video_thread.join()
    time.sleep(0.5)  # Ensure device release

def start_single(output_file, timer=6, fps=30):
    def led_light():
        RED_PIN = 26
        GREEN_PIN = 19
        BLUE_PIN = 13
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(RED_PIN, GPIO.OUT)
            GPIO.setup(GREEN_PIN, GPIO.OUT)
            GPIO.setup(BLUE_PIN, GPIO.OUT)

            def turn_on_led(red, green, blue):
                GPIO.output(RED_PIN, not red)
                GPIO.output(GREEN_PIN, not green)
                GPIO.output(BLUE_PIN, not blue)

            print("Turning on Red for 6 seconds")
            turn_on_led(GPIO.LOW, GPIO.LOW, GPIO.HIGH)
            time.sleep(timer)
        except Exception as e:
            print(f"GPIO error: {e}")
            logging.error(f"GPIO error: {e}")
        finally:
            try:
                turn_on_led(GPIO.HIGH, GPIO.HIGH, GPIO.HIGH)
                GPIO.cleanup()
            except Exception as e:
                print(f"Error cleaning up GPIO: {e}")
                logging.error(f"Error cleaning up GPIO: {e}")

    def record_video():
        command = [
            "ffmpeg",
            "-y",
            "-f", "v4l2",
            "-video_size", "1920x1080",
            "-input_format", "mjpeg",
            "-i", "/dev/video0",
            "-c:v", "copy",
            "-t", str(timer),
            "-r", str(fps),
            output_file
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            print("Ok:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            logging.error(f"FFmpeg error: {e.stderr}")
        except Exception as e:
            print(f"Exception in record_video: {e}")
            logging.error(f"Exception in record_video: {e}")

    check_device()
    led_thread = threading.Thread(target=led_light)
    video_thread = threading.Thread(target=record_video)
    led_thread.start()
    video_thread.start()
    led_thread.join()
    video_thread.join()
    time.sleep(0.5)  # Ensure device release

def generate_csv(video_path, csv_path, name):
    video_name = name
    output_paths = {}
    os.makedirs(csv_path, exist_ok=True)
    video_file = os.path.join(video_path, f"{name}.mkv")

    all_exist = True
    segments = ['red', 'orange', 'yellow']
    for segment in segments:
        output_path = os.path.join(csv_path, f"{video_name}_{segment}_flattened.csv")
        output_paths[segment] = output_path
        if not os.path.exists(output_path):
            all_exist = False
    if all_exist:
        return video_name, output_paths

    if not os.path.exists(video_file):
        print(f"[Warning] Video not found: {video_file}")
        logging.error(f"Video not found: {video_file}")
        return video_name, None
    if not os.access(video_file, os.R_OK):
        print(f"Error: No read permission for {video_file}")
        logging.error(f"No read permission for {video_file}")
        return video_name, None

    try:
        video = cv2.VideoCapture(video_file)
        if not video.isOpened():
            print(f"[Error] Cannot open video: {video_file}")
            logging.error(f"Cannot open video: {video_file}")
            return video_name, None

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_rate <= 0 or total_frames == 0:
            print(f"[Error] Invalid frame rate ({frame_rate}) or total frames ({total_frames}) for {video_name}")
            logging.error(f"Invalid frame rate or total frames for {video_name}")
            return video_name, None

        roi_size = min(frame_width, frame_height) // 3
        if roi_size <= 0:
            print(f"[Error] Invalid ROI size for {video_name}: frame_width={frame_width}, frame_height={frame_height}")
            logging.error(f"Invalid ROI size for {video_name}")
            return video_name, None

        roi_top = (frame_height - roi_size) // 2
        roi_left = (frame_width - roi_size) // 2
        roi_bottom = roi_top + roi_size
        roi_right = roi_left + roi_size

        frames_per_segment = int(2 * frame_rate)
        segment_ranges = {
            'red': (0, frames_per_segment),
            'orange': (frames_per_segment, 2 * frames_per_segment),
            'yellow': (2 * frames_per_segment, min(3 * frames_per_segment, total_frames))
        }

        for segment, (segment_start, segment_end) in segment_ranges.items():
            segment_end = min(segment_end, total_frames)
            segment_start = min(segment_start, segment_end)
            temp_csv = os.path.join(csv_path, f"{video_name}_{segment}_temp.csv")
            try:
                with open(temp_csv, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Frame", "R", "G", "B", "H", "S", "V", "L", "A", "Grayscale"])
                    frame_count = 0
                    video.set(cv2.CAP_PROP_POS_FRAMES, segment_start)

                    while frame_count < segment_end - segment_start:
                        ret, frame = video.read()
                        if not ret:
                            break
                        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
                        roi_lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
                        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
                        r, g, b = cv2.split(roi_rgb)
                        h, s, v = cv2.split(roi_hsv)
                        l, a, _ = cv2.split(roi_lab)
                        for i in range(len(r)):
                            writer.writerow([
                                frame_count + segment_start,
                                r[i][0], g[i][0], b[i][0],
                                h[i][0], s[i][0], v[i][0],
                                l[i][0], a[i][0], roi_gray[i][0]
                            ])
                        frame_count += 1

                df = pd.read_csv(temp_csv) if os.path.exists(temp_csv) else pd.DataFrame()
                channels = ['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'Grayscale']
                histograms = {}
                for channel in channels:
                    channel_histograms = []
                    for frame in range(segment_start, segment_end):
                        filtered_df = df[df['Frame'] == frame]
                        if not filtered_df.empty:
                            hist, _ = np.histogram(filtered_df[channel], bins=256, range=(0, 256), density=True)
                            channel_histograms.append(hist)
                    histograms[channel] = np.mean(channel_histograms, axis=0) if channel_histograms else np.zeros(256)
                histogram_df = pd.DataFrame(histograms)
                flattened = histogram_df.to_numpy().reshape(1, -1, order='F')
                output_path = os.path.join(csv_path, f"{video_name}_{segment}_flattened.csv")
                pd.DataFrame(flattened).to_csv(output_path, index=False, header=False)
                output_paths[segment] = output_path
            finally:
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
        return video_name, output_paths
    except Exception as e:
        print(f"[Error] Failed on {video_name}: {e}")
        logging.error(f"Failed on {video_name}: {e}")
        return video_name, None
    finally:
        video.release()

def predict_lite(model, name, csv_path):
    segments = ['red', 'orange', 'yellow']
    csv_paths = [os.path.join(csv_path, f"{name}_{segment}_flattened.csv") for segment in segments]
    data_tensors = []
    expected_size = 2304
    for csv_file, segment in zip(csv_paths, segments):
        if not os.path.exists(csv_file):
            print(f"[Error] CSV file not found for {segment} segment: {csv_file}")
            logging.error(f"CSV file not found for {segment} segment: {csv_file}")
            raise FileNotFoundError(f"[Error] CSV file not found for {segment} segment: {csv_file}")
        if os.path.getsize(csv_file) == 0:
            print(f"[Error] Empty CSV file: {csv_file}")
            logging.error(f"Empty CSV file: {csv_file}")
            raise ValueError(f"[Error] Empty CSV file: {csv_file}")
        try:
            data = pd.read_csv(csv_file, header=None).to_numpy()
            if data.shape != (1, expected_size):
                print(f"[Error] Invalid shape for {segment} CSV {csv_file}: expected (1, {expected_size}), got {data.shape}")
                logging.error(f"Invalid shape for {segment} CSV {csv_file}: expected (1, {expected_size}), got {data.shape}")
                raise ValueError(f"[Error] Invalid shape for {segment} CSV")
            data_tensors.append(data)
        except Exception as e:
            print(f"[Error] Failed to load {segment} CSV {csv_file}: {e}")
            logging.error(f"Failed to load {segment} CSV {csv_file}: {e}")
            raise RuntimeError(f"[Error] Failed to load {segment} CSV")
    x_red = torch.tensor(data_tensors[0], dtype=torch.float32)
    x_orange = torch.tensor(data_tensors[1], dtype=torch.float32)
    x_yellow = torch.tensor(data_tensors[2], dtype=torch.float32)
    device = next(model.parameters()).device
    x_red = x_red.to(device)
    x_orange = x_orange.to(device)
    x_yellow = x_yellow.to(device)
    model.eval()
    with torch.no_grad():
        output = model(x_red, x_orange, x_yellow)
        pred = output.item()
    print(f"Predicted Hemoglobin: {pred:.2f} g/dL")
    return float(pred)

def display_on_lcd(pred, lcd):
    def safe_exit(signum, frame):
        exit(1)
    signal(SIGTERM, safe_exit)
    signal(SIGHUP, safe_exit)
    try:
        lcd.clear()
        lcd.text("Your Hb level is,", 1)
        lcd.text(f"{pred:.2f}", 2)
        time.sleep(15)
    except OSError as e:
        print(f"I2C error in display_on_lcd: {e}")
        logging.error(f"I2C error in display_on_lcd: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            lcd.clear()
        except OSError as e:
            print(f"I2C error on clear in display_on_lcd: {e}")
            logging.error(f"I2C error on clear in display_on_lcd: {e}")

def display(msg, lcd):
    try:
        lcd.right_string(f"{msg}")
    except OSError as e:
        print(f"I2C error in display: {e}")
        logging.error(f"I2C error in display: {e}")