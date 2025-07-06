#!/home/nishad/Nishad_env/bin/python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os,cv2,csv,itertools,time,subprocess
from RPLCD import CharLCD
import RPi.GPIO as GPIO
import time
from signal import signal, SIGTERM, SIGHUP, pause
from rpi_lcd import LCD
import time
import shlex
#from AllinOne import predict_lite
import os
import torch.nn.functional as F
os.chdir("/home/nishad/Nishad/")

def unique_file(basename, ext="mkv"):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    name=basename.split("/")[-1]
    k=""
    while os.path.exists(actualname):
        k=next(c)
        actualname = "%s%d.%s" % (basename,k, ext)
    name=name+f"{k}"
    return name,actualname

import cv2
import numpy as nps
import subprocess
import threading
import time
import RPi.GPIO as GPIO

import threading
import time
import subprocess
import RPi.GPIO as GPIO

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

# ======== Updated Model Architecture ======== #
class DeepHbNetFlexible(nn.Module):
    def __init__(self, input_size=2304, dropout_rate=0.2, num_blocks=4):
        super(DeepHbNetFlexible, self).__init__()
        
        # Three parallel branches for red, orange, yellow CSVs
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

        # Combine the outputs of the three branches
        combined_size = 1024 * 3  # Concatenate outputs from three branches
        hidden_sizes = [1024,512, 256, 128]
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
        # Process each CSV through its respective branch
        red_out = self.red_branch(x_red)
        orange_out = self.orange_branch(x_orange)
        yellow_out = self.yellow_branch(x_yellow)
        
        # Concatenate the outputs
        x = torch.cat((red_out, orange_out, yellow_out), dim=1)
        
        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)
        
        return self.output(x)

def start_dual(output_file, timer=2, fps=30):

    def led_light():
        # Define the GPIO pins for red, green, and blue
        RED_PIN = 26
        GREEN_PIN = 19
        BLUE_PIN = 13
        FREQ = 100
        # Setup GPIO mode
        GPIO.setmode(GPIO.BCM)  # Using BCM numbering (GPIO numbers)
        GPIO.setwarnings(False)  # Disable GPIO warnings

        # Setup the GPIO pins as output
        GPIO.setup(RED_PIN, GPIO.OUT)
        GPIO.setup(GREEN_PIN, GPIO.OUT)
        GPIO.setup(BLUE_PIN, GPIO.OUT)
        
        red_pwm = GPIO.PWM(RED_PIN, FREQ)
        green_pwm = GPIO.PWM(GREEN_PIN, FREQ)
        blue_pwm = GPIO.PWM(BLUE_PIN, FREQ)
        
        red_pwm.start(0)
        green_pwm.start(0)
        blue_pwm.start(0)
        
        def set_color(red_pwm, green_pwm, blue_pwm,r,g,b):
            red_pwm.ChangeDutyCycle(r)
            green_pwm.ChangeDutyCycle(g)
            blue_pwm.ChangeDutyCycle(b)
        
        
        def turn_on_led(red, green, blue):
#             GPIO.output(RED_PIN, not red)
#             GPIO.output(GREEN_PIN, not green)
#             GPIO.output(BLUE_PIN, not blue)
            
            GPIO.output(RED_PIN, red)
            GPIO.output(GREEN_PIN,green)
            GPIO.output(BLUE_PIN, blue)

        try:
            print("Turning on Red for 6 seconds")
#             turn_on_led(GPIO.LOW, GPIO.LOW, GPIO.HIGH)  # Turn on Yellow (red + green)
#             turn_on_led(GPIO.HIGH, GPIO.LOW, GPIO.LOW)  # Turn on Red (red )
#             time.sleep(timer)  # Use the timer parameter for sleep duration
# #           
#             turn_on_led(GPIO.HIGH, GPIO.HIGH, GPIO.LOW)  # Turn on Red (red )
#             set_color(red_pwm, green_pwm, blue_pwm,r,g,b)
#             time.sleep(timer)  # Use the timer parameter for sleep duration
#             turn_on_led(GPIO.HIGH, GPIO.LOW, GPIO.LOW)  # Turn on Red (red )
#             time.sleep(timer)  # Use the timer parameter for sleep duration
            set_color(red_pwm, green_pwm, blue_pwm,100,0,0)
            time.sleep(timer)
            set_color(red_pwm, green_pwm, blue_pwm,100,25,0)
            time.sleep(timer)
            set_color(red_pwm, green_pwm, blue_pwm,100,65,0)
            time.sleep(timer)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Turn off all LEDs
            turn_on_led(GPIO.HIGH, GPIO.HIGH, GPIO.HIGH)
                
            red_pwm.stop()
            green_pwm.stop()
            blue_pwm.stop()
        
            # Cleanup GPIO settings
            GPIO.cleanup()

    def record_video():
        command = f"ffmpeg -f v4l2 -video_size 1920x1080 -input_format mjpeg -i /dev/video0 -c:v copy -t {timer*3} -r {fps} {output_file}"
        status_output = subprocess.getstatusoutput(command)
        print(status_output)
        if status_output[0] == 0:
            print("Ok:", status_output[1])
        else:
            print("Error:", status_output[1])

    # Create threads for led_light and record_video functions
    led_thread = threading.Thread(target=led_light)
    video_thread = threading.Thread(target=record_video)

    # Start both threads
    led_thread.start()
    video_thread.start()

    # Wait for both threads to complete
    led_thread.join()
    video_thread.join()

def start_single(output_file, timer=6, fps=30):
    
    def led_light():
        # Define the GPIO pins for red, green, and blue
        RED_PIN = 26
        GREEN_PIN = 19
        BLUE_PIN = 13

        # Setup GPIO mode
        GPIO.setmode(GPIO.BCM)  # Using BCM numbering (GPIO numbers)
        GPIO.setwarnings(False)  # Disable GPIO warnings

        # Setup the GPIO pins as output
        GPIO.setup(RED_PIN, GPIO.OUT)
        GPIO.setup(GREEN_PIN, GPIO.OUT)
        GPIO.setup(BLUE_PIN, GPIO.OUT)

        def turn_on_led(red, green, blue):
            GPIO.output(RED_PIN, not red)
            GPIO.output(GREEN_PIN, not green)
            GPIO.output(BLUE_PIN, not blue)

        try:
            print("Turning on Red for 30 seconds")
            turn_on_led(GPIO.LOW, GPIO.LOW, GPIO.HIGH)  # Turn on Red
            time.sleep(timer)  # Use the timer parameter for sleep duration
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Turn off all LEDs
            turn_on_led(GPIO.HIGH, GPIO.HIGH, GPIO.HIGH)
            # Cleanup GPIO settings
            GPIO.cleanup()

        def record_video():
            command = [
                "ffmpeg",
                "-y",  # Overwrite without asking
                "-f", "v4l2",
                "-video_size", "1920x1080",
                "-input_format", "mjpeg",
                "-i", "/dev/video0",
                "-c:v", "copy",
                "-t", str(timer),  # in seconds
                "-r", str(fps),
                output_file
            ]

            try:
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0:
                    print("Ok:", result.stdout)
                else:
                    print("Error:", result.stderr)
            except Exception as e:
                print(f"An exception occurred: {e}")


    # Create threads for led_light and record_video functions
    led_thread = threading.Thread(target=led_light)
    video_thread = threading.Thread(target=record_video)

    # Start both threads
    led_thread.start()
    video_thread.start()

    # Wait for both threads to complete
    led_thread.join()
    video_thread.join()



#def start(output_file,timer=30,fps=30):
    
 #   command=f"parallel --lb ::: 'ffmpeg -f v4l2 -video_size 1920x1080 -input_format mjpeg -i /dev/video2 -c:v copy -t {timer} -r {fps} {output_file}'"
  #  status_output = subprocess.getstatusoutput(command)
   # print(status_output)
    #if status_output[0] == 0:
     #   print("Ok:", status_output[1])

def display(msg):
    lcd=CharLCD(cols=20,rows=2,pin_rs=37,pin_e=35,pins_data=[33,31,29,23])
    lcd.right_string(f"{msg}")

def generate_csv(video_path, csv_path, name):
    video_name = name  # Use provided name for file naming
    output_paths = {}

    # Ensure csv_path directory exists
    os.makedirs(csv_path, exist_ok=True)

    # Construct full video file path
    video_file = os.path.join(video_path, f"{name}.mkv")

    # Check if all segment CSVs exist
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
        return video_name, None

    try:
        video = cv2.VideoCapture(video_file)
        if not video.isOpened():
            print(f"[Error] Cannot open video: {video_file}")
            return video_name, None

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate frame rate and total frames
        if frame_rate <= 0 or total_frames == 0:
            print(f"[Error] Invalid frame rate ({frame_rate}) or total frames ({total_frames}) for {video_name}")
            return video_name, None

        # Validate ROI size
        roi_size = min(frame_width, frame_height) // 3
        if roi_size <= 0:
            print(f"[Error] Invalid ROI size for {video_name}: frame_width={frame_width}, frame_height={frame_height}")
            return video_name, None

        roi_top = (frame_height - roi_size) // 2
        roi_left = (frame_width - roi_size) // 2
        roi_bottom = roi_top + roi_size
        roi_right = roi_left + roi_size

        # Define segment ranges for 6-second video (2 seconds each)
        frames_per_segment = int(2 * frame_rate)  # 2 seconds per segment
        segment_ranges = {
            'red': (0, frames_per_segment),                    # 0-2 seconds
            'orange': (frames_per_segment, 2 * frames_per_segment),  # 2-4 seconds
            'yellow': (2 * frames_per_segment, min(3 * frames_per_segment, total_frames))  # 4-6 seconds
        }

        for segment, (segment_start, segment_end) in segment_ranges.items():
            # Ensure segment_end does not exceed total_frames
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

                        # Write first pixel of each row in ROI
                        for i in range(len(r)):
                            writer.writerow([
                                frame_count + segment_start,
                                r[i][0], g[i][0], b[i][0],
                                h[i][0], s[i][0], v[i][0],
                                l[i][0], a[i][0], roi_gray[i][0]
                            ])
                        frame_count += 1

                # Process histograms
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
                # Clean up temporary CSV file
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)

        return video_name, output_paths

    except Exception as e:
        print(f"[Error] Failed on {video_name}: {e}")
        return video_name, None
    finally:
        video.release()



def predict_lite(model, name, csv_path):
    segments = ['red', 'orange', 'yellow']
    csv_paths = [os.path.join(csv_path, f"{name}_{segment}_flattened.csv") for segment in segments]
    
    # Load and validate CSV files
    data_tensors = []
    expected_size = 2304  # Expected size of each CSV (9 channels × 256 bins)
    for csv_file, segment in zip(csv_paths, segments):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"[Error] CSV file not found for {segment} segment: {csv_file}")
        try:
            data = pd.read_csv(csv_file, header=None).to_numpy()
            if data.shape != (1, expected_size):
                raise ValueError(f"[Error] Invalid shape for {segment} CSV {csv_file}: expected (1, {expected_size}), got {data.shape}")
            data_tensors.append(data)
        except Exception as e:
            raise RuntimeError(f"[Error] Failed to load {segment} CSV {csv_file}: {e}")

    # Convert to PyTorch tensors
    x_red = torch.tensor(data_tensors[0], dtype=torch.float32)
    x_orange = torch.tensor(data_tensors[1], dtype=torch.float32)
    x_yellow = torch.tensor(data_tensors[2], dtype=torch.float32)

    # Move tensors to the model's device
    device = next(model.parameters()).device
    x_red = x_red.to(device)
    x_orange = x_orange.to(device)
    x_yellow = x_yellow.to(device)

    # Model prediction
    model.eval()
    with torch.no_grad():
        output = model(x_red, x_orange, x_yellow)
        pred = output.item()

    print(f"Predicted Hemoglobin: {pred:.2f} g/dL")

    return float(pred)

def led_light():
    # Define the GPIO pins for red, green, and blue
    RED_PIN = 26
    GREEN_PIN = 19
    BLUE_PIN = 13

    # Setup GPIO mode
    GPIO.setmode(GPIO.BCM)  # Using BCM numbering (GPIO numbers)
    GPIO.setwarnings(False)  # Disable GPIO warnings

    # Setup the GPIO pins as output
    GPIO.setup(RED_PIN, GPIO.OUT)
    GPIO.setup(GREEN_PIN, GPIO.OUT)
    GPIO.setup(BLUE_PIN, GPIO.OUT)

    def turn_on_led(red, green, blue):
        GPIO.output(RED_PIN, not red)
        GPIO.output(GREEN_PIN, not green)
        GPIO.output(BLUE_PIN, not blue)

    try:
        print("Turning on Red for 6 seconds")
        turn_on_led(GPIO.LOW, GPIO.LOW, GPIO.HIGH) 
        time.sleep(30)


    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Turn off all LEDs
        turn_on_led(GPIO.HIGH, GPIO.HIGH, GPIO.HIGH)
        # Cleanup GPIO settings
        GPIO.cleanup()


def display_on_lcd(pred):
    lcd = LCD()

    def safe_exit(signum, frame):
        exit(1)
        
    signal(SIGTERM, safe_exit)
    signal(SIGHUP, safe_exit)

    try:
        lcd.text("Your Hb level is,", 1)
        lcd.text("{:2f}".format(pred), 2)

        time.sleep(15)

    except KeyboardInterrupt:
        pass

    finally:
        lcd.clear()