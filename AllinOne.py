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
from model import DeepHbNetFlexible as DeepHbNet

def start_dual(output_file, timer=10, fps=30):

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
            print("Turning on Red for 30 seconds")
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
        command = f"ffmpeg -f v4l2 -video_size 1920x1080 -input_format mjpeg -i /dev/video3 -c:v copy -t {timer*3} -r {fps} {output_file}"
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

def start_single(output_file, timer=30, fps=30):
    
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
                "-i", "/dev/video3",
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
    lcd=CharLCD(cols=16,rows=2,pin_rs=37,pin_e=35,pins_data=[40,38,36,32,33,31,29,23])
    lcd.right_string(f"{msg}")

def generate_csv(video_filename, csv_filename):
    video = cv2.VideoCapture(video_filename)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_size = min(frame_width, frame_height) // 3
    roi_top = (frame_height - roi_size) // 2
    roi_left = (frame_width - roi_size) // 2
    roi_bottom = roi_top + roi_size
    roi_right = roi_left + roi_size

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "R", "G", "B", "H", "S", "V", "L", "A", "B", "Grayscale"])
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            roi = frame[roi_top:roi_bottom, roi_left:roi_right]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            r, g, b = cv2.split(roi_rgb)
            h, s, v = cv2.split(roi_hsv)
            l, a, b = cv2.split(roi_lab)

            for i in range(len(r)):
                writer.writerow([
                    frame_count,
                    r[i][0], g[i][0], b[i][0],
                    h[i][0], s[i][0], v[i][0],
                    l[i][0], a[i][0], b[i][0],
                    roi_gray[i][0]
                ])
            frame_count += 1
    video.release()
    print("Created CSV Successfully !")


def generate_average_histogram(csv_filename, output_filename):
    df = pd.read_csv(csv_filename)
    histograms = {}
    channels = ['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'B', 'Grayscale']

    for channel in channels:
        channel_histograms = []

        for frame in range(50, 550):
            filtered_df = df[df['Frame'] == frame]
            hist, _ = np.histogram(filtered_df[channel], bins=256, range=(0, 256), density=True)
            channel_histograms.append(hist)

        average_histogram = np.mean(channel_histograms, axis=0)
        histograms[channel] = average_histogram

    histogram_df = pd.DataFrame(histograms)
    histogram_df.to_csv(output_filename, index=False)
    print("Created Histogram Successfully !")


def load_lite_model(path):
  model = DeepHbNet(num_blocks=4)
  model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
  model.eval()
  return model


def preprocess(df):
    matrix_test = np.zeros((1, 2304))
    for i in range(1):
        for j, col in enumerate(['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'Grayscale']):
            column_data = df[col].values
            matrix_test[i, j * 256:(j + 1) * 256] = column_data[:256]

    return matrix_test

def predict_lite(model, csv_file):
    print(csv_file)
    data = pd.read_csv(csv_file, header=None,skiprows=1).values.astype(np.float32)
    data = data.reshape(-1)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    input_tensor = torch.tensor(data, dtype=torch.float32)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.item()
    anemia_threshold = 11.0
    status = "Anemic" if pred < anemia_threshold else "Not Anemic"
    print(pred)
    print(f"Status: {status}")

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
        print("Turning on Red for 30 seconds")
        turn_on_led(GPIO.LOW, GPIO.LOW, GPIO.HIGH)  # Turn on Yellow (red + green)
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
