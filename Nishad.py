#!/home/nishad/Nishad_env/bin/python

import os
import time
import sys
from signal import signal, SIGTERM, SIGHUP
from AllinOne import *
from rpi_lcd import LCD
import torch
import logging

# Setup logging
logging.basicConfig(filename='nishad.log', level=logging.ERROR)

# 🕒 Start measuring total time
start_time = time.time()

vid_path = "Data/vids/"
csv_path = "Data/csv_data/"
hist_path = "Data/histogram/"
paths = [vid_path, csv_path, hist_path]

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
        if not os.access(p, os.W_OK):
            print(f"Error: No write permission for directory: {p}")
            logging.error(f"No write permission for directory: {p}")
            exit(1)

def ensure_writable_path(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.access(dir_path, os.W_OK):
        print(f"Error: Cannot write to directory: {dir_path}")
        logging.error(f"Cannot write to directory: {dir_path}")
        exit(1)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting existing file {file_path}: {e}")
            logging.error(f"Error deleting existing file {file_path}: {e}")
            exit(1)

def display_greet(lcd):
    signal(SIGTERM, lambda s, f: exit(1))
    signal(SIGHUP, lambda s, f: exit(1))
    try:
        lcd.clear()
        lcd.text("Lets Start", 1)
        lcd.text("the procedure", 2)
        time.sleep(1)
    except OSError as e:
        print(f"I2C error in display_greet: {e}")
        logging.error(f"I2C error in display_greet: {e}")
    except KeyboardInterrupt:
        pass

def display_waiting(lcd):
    signal(SIGTERM, lambda s, f: exit(1))
    signal(SIGHUP, lambda s, f: exit(1))
    try:
        lcd.clear()
        lcd.text("Please dont move,", 1)
        lcd.text("for 6 seconds", 2)
        time.sleep(5)
    except OSError as e:
        print(f"I2C error in display_waiting: {e}")
        logging.error(f"I2C error in display_waiting: {e}")
    except KeyboardInterrupt:
        pass

def display_successful(lcd):
    signal(SIGTERM, lambda s, f: exit(1))
    signal(SIGHUP, lambda s, f: exit(1))
    try:
        lcd.clear()
        lcd.text("Reading complete", 1)
        lcd.text("Remove finger", 2)
        time.sleep(2)
    except OSError as e:
        print(f"I2C error in display_successful: {e}")
        logging.error(f"I2C error in display_successful: {e}")
    except KeyboardInterrupt:
        pass

def display_ongoing(lcd):
    signal(SIGTERM, lambda s, f: exit(1))
    signal(SIGHUP, lambda s, f: exit(1))
    try:
        lcd.clear()
        lcd.text("Computation ", 1)
        lcd.text("Ongoing", 2)
        time.sleep(2)
    except OSError as e:
        print(f"I2C error in display_ongoing: {e}")
        logging.error(f"I2C error in display_ongoing: {e}")
    except KeyboardInterrupt:
        pass

def display_next(lcd):
    signal(SIGTERM, lambda s, f: exit(1))
    signal(SIGHUP, lambda s, f: exit(1))
    try:
        lcd.clear()
        lcd.text("For new reading", 1)
        lcd.text("Wait 10 second", 2)
        time.sleep(10)
    except OSError as e:
        print(f"I2C error in display_next: {e}")
        logging.error(f"I2C error in display_next: {e}")
    except KeyboardInterrupt:
        pass

# Change directory and prepare paths
os.chdir("/home/hbmeter/Hb_meter/")
ensure_dirs(paths)

name, record_path = unique_file(vid_path + "record")
csv_file = csv_path + name + ".csv"

ensure_writable_path(csv_file)

# Initialize LCD once
lcd = LCD(address=0x27)  # Replace with your I2C address if different
try:
    display_greet(lcd)
    print(record_path)
    display_waiting(lcd)

    # Start dual is used to turn on LED and record video
    try:
        start_dual(record_path)
    except Exception as e:
        print(f"Error in start_dual: {e}")
        logging.error(f"Error in start_dual: {e}")
        exit(1)

    display_successful(lcd)

    # ➕ Show LCD update: CSV Generation
    try:
        lcd.clear()
        lcd.text("Processing...", 1)
        lcd.text("Please wait", 2)
    except OSError as e:
        print(f"I2C error during processing display: {e}")
        logging.error(f"I2C error during processing display: {e}")

    try:
        generate_csv(vid_path, csv_path, name)
    except PermissionError as e:
        print(f"Permission denied when writing {csv_file}: {e}")
        logging.error(f"Permission denied when writing {csv_file}: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error during CSV generation: {e}")
        logging.error(f"Unexpected error during CSV generation: {e}")
        exit(1)

    # ➕ Show LCD update: Model Loading
    try:
        lcd.clear()
        lcd.text("Loading Model", 1)
        lcd.text("Please wait...", 2)
    except OSError as e:
        print(f"I2C error during model loading display: {e}")
        logging.error(f"I2C error during model loading display: {e}")

    MODEL_PATH = 'best_model_segments.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepHbNetFlexible().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        logging.error(f"Error loading model: {e}")
        exit(1)

    # ➕ Show LCD update: Prediction
    try:
        lcd.clear()
        lcd.text("Predicting Hb", 1)
        lcd.text("In Progress...", 2)
    except OSError as e:
        print(f"I2C error during prediction display: {e}")
        logging.error(f"I2C error during prediction display: {e}")

    try:
        pred = predict_lite(model, name, csv_path)
    except Exception as e:
        print(f"Error during prediction: {e}")
        logging.error(f"Error during prediction: {e}")
        exit(1)

    # ➕ Show prediction on LCD
    try:
        display_on_lcd(pred, lcd)
    except Exception as e:
        print(f"Error in display_on_lcd: {e}")
        logging.error(f"Error in display_on_lcd: {e}")
        exit(1)

    # Final message
    display_next(lcd)

finally:
    try:
        lcd.clear()
    except OSError as e:
        print(f"I2C error on final clear: {e}")
        logging.error(f"I2C error on final clear: {e}")

# 🕒 End measuring total time
end_time = time.time()
total_time = end_time - start_time
mins, secs = divmod(total_time, 60)
print(f"⏱️ Total time taken: {int(mins)} min {secs:.1f} sec")

# Optional: Next process
# subprocess.run(["python3", "/home/nishad/Nishad/nishad_switch.py"])