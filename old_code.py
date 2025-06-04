#!/home/nishad/Nishad_env/bin/python

from tflite_runtime.interpreter import Interpreter 
import pandas as pd
import numpy as np
import os,cv2,csv,itertools,time,subprocess
from RPLCD import CharLCD
import RPi.GPIO as GPIO
import time
from signal import signal, SIGTERM, SIGHUP, pause
from rpi_lcd import LCD
import time
#from AllinOne import predict_lite
import os

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

def start_dual(output_file, timer=30, fps=30):

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
            time.sleep(timer)  # Use the timer parameter for sleep duration

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Turn off all LEDs
            turn_on_led(GPIO.HIGH, GPIO.HIGH, GPIO.HIGH)
            # Cleanup GPIO settings
            GPIO.cleanup()

    def record_video():
        command = f"ffmpeg -f v4l2 -video_size 1920x1080 -input_format mjpeg -i /dev/video3 -c:v copy -t {timer} -r {fps} {output_file}"
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
        # Use the rpicam-vid command to record video
        try:
            print("Starting video recording...")
            subprocess.run(['rpicam-vid', '-t', str(timer * 1000), '-o', output_file, '-f'])
            print("Recording complete.")
        except Exception as e:
            print(f"An error occurred during recording: {e}")

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
    interpreter=Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess(df):
    matrix_test = np.zeros((1, 2304))
    for i in range(1):
        for j, col in enumerate(['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'Grayscale']):
            column_data = df[col].values
            matrix_test[i, j * 256:(j + 1) * 256] = column_data[:256]

    return matrix_test

def predict_lite(model,file):
    test_data = pd.read_csv(file)
    matrix_test=preprocess(test_data).astype(np.float32)
    input_details=model.get_input_details()
    output_details=model.get_output_details()

    model.set_tensor(input_details[0]['index'],matrix_test)
    model.invoke()
    prediction=model.get_tensor(output_details[0]['index'])
    print(prediction)
    pred = prediction[0][0]
    anemia_threshold = 11.0
    anemia_predictions_test = np.where(prediction < anemia_threshold, 'Anemic', 'Not Anemic')

    """with open("results.txt",'w+') as f:
        f.writelines('Predictions\tAnemia \n')
        for i in range(len(prediction)):
            f.writelines('{:.2f}\t\t{} \n'.format(prediction[i, 0], anemia_predictions_test[i]))
        f.close()"""
    return(pred)
    print('Predictions\tAnemia')
    for i in range(len(prediction)):
        print('{:.2f}\t\t{}'.format(prediction[i, 0], anemia_predictions_test[i]))
    

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


#!/home/nishad/Nishad_env/bin/python

import os

os.chdir("/home/nishad/Nishad/")

from AllinOne import *

vid_path = "Data/vids/"
csv_path = "Data/csv_data/"
hist_path = "Data/histogram/"
paths = [vid_path, csv_path, hist_path]

def display_greet():
    lcd = LCD()

    def safe_exit(signum, frame):
        exit(1)
        
    signal(SIGTERM, safe_exit)
    signal(SIGHUP, safe_exit)

    try:
        lcd.clear()
        lcd.text("Lets Start", 1)
        lcd.text("the procedure", 2)
        time.sleep(5)

    except KeyboardInterrupt:
        pass

    finally:
        lcd.clear()

def display_waiting():
    lcd = LCD()

    def safe_exit(signum, frame):
        exit(1)
        
    signal(SIGTERM, safe_exit)
    signal(SIGHUP, safe_exit)

    try:
        lcd.clear()
        lcd.text("Please dont move,", 1)
        lcd.text("for 30 seconds", 2)
        time.sleep(5)

    except KeyboardInterrupt:
        pass

    finally:
        lcd.clear()
    
def display_successful():
    lcd = LCD()

    def safe_exit(signum, frame):
        exit(1)
        
    signal(SIGTERM, safe_exit)
    signal(SIGHUP, safe_exit)

    try:
        lcd.clear()
        lcd.text("Reading complete", 1)
        lcd.text("Remove finger", 2)
        time.sleep(5)

    except KeyboardInterrupt:
        pass

    finally:
        lcd.clear()
        
def display_ongoing():
    lcd = LCD()

    def safe_exit(signum, frame):
        exit(1)
        
    signal(SIGTERM, safe_exit)
    signal(SIGHUP, safe_exit)

    try:
        lcd.clear()
        lcd.text("Computation ", 1)
        lcd.text("Ongoing", 2)
        time.sleep(5)

    except KeyboardInterrupt:
        pass

    finally:
        lcd.clear()
        
def display_next():
    lcd = LCD()

    def safe_exit(signum, frame):
        exit(1)
        
    signal(SIGTERM, safe_exit)
    signal(SIGHUP, safe_exit)

    try:
        lcd.clear()
        lcd.text("For new reading", 1)
        lcd.text("Wait 10 second", 2)
        time.sleep(10)

    except KeyboardInterrupt:
        pass

    finally:
        lcd.clear()
        
for i in paths:
    if not os.path.exists(i):
        os.makedirs(i)

output_name, record_path = unique_file(vid_path + "record")
csv_file = csv_path + output_name + ".csv"
avg_file = hist_path + output_name + "avg" + ".csv"

display_greet()
print(record_path)
display_waiting()
start_dual(record_path)
display_successful()
generate_csv(record_path, csv_file)
display_ongoing()
generate_average_histogram(csv_file, avg_file)

model = load_lite_model("/home/nishad/Nishad/model_quant.tflite")
pred = predict_lite(model, avg_file)
display_on_lcd(pred)

display_next()

#subprocess.run(["python3", "/home/nishad/Nishad/nishad_switch.py"])


