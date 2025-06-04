#!/home/nishad/Nishad_env/bin/python

import os
import time
from signal import signal, SIGTERM, SIGHUP
from AllinOne import *

vid_path = "Data/vids/"
csv_path = "Data/csv_data/"
hist_path = "Data/histogram/"
paths = [vid_path, csv_path, hist_path]

def ensure_dirs(paths):
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
        # Ensure writable
        if not os.access(p, os.W_OK):
            print(f"Error: No write permission for directory: {p}")
            exit(1)

def ensure_writable_path(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.access(dir_path, os.W_OK):
        print(f"Error: Cannot write to directory: {dir_path}")
        exit(1)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting existing file {file_path}: {e}")
            exit(1)

def display_greet():
    lcd = LCD()
    signal(SIGTERM, lambda s, f: exit(1))
    signal(SIGHUP, lambda s, f: exit(1))
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
    signal(SIGTERM, lambda s, f: exit(1))
    signal(SIGHUP, lambda s, f: exit(1))
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
    signal(SIGTERM, lambda s, f: exit(1))
    signal(SIGHUP, lambda s, f: exit(1))
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
    signal(SIGTERM, lambda s, f: exit(1))
    signal(SIGHUP, lambda s, f: exit(1))
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
    signal(SIGTERM, lambda s, f: exit(1))
    signal(SIGHUP, lambda s, f: exit(1))
    try:
        lcd.clear()
        lcd.text("For new reading", 1)
        lcd.text("Wait 10 second", 2)
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        lcd.clear()

# Change directory and prepare paths
os.chdir("/home/nishad/Nishad/")
ensure_dirs(paths)

output_name, record_path = unique_file(vid_path + "record")
csv_file = csv_path + output_name + ".csv"
avg_file = hist_path + output_name + "avg_.csv"

ensure_writable_path(csv_file)
ensure_writable_path(avg_file)

display_greet()
print(record_path)
display_waiting()
# Start dual is used to turn on LED
start_dual(record_path)
display_successful()

try:
    generate_csv(record_path, csv_file)
except PermissionError as e:
    print(f"Permission denied when writing {csv_file}: {e}")
    exit(1)
except Exception as e:
    print(f"Unexpected error during CSV generation: {e}")
    exit(1)

display_ongoing()

try:
    print(avg_file)
    generate_average_histogram(csv_file, avg_file)
except Exception as e:
    print(f"Error generating histogram: {e}")
    exit(1)


model = load_lite_model("/home/nishad/Nishad/best_model_99.pth")
pred = predict_lite(model, avg_file)
display_on_lcd(pred)


display_next()

# Optionally trigger the next process
# subprocess.run(["python3", "/home/nishad/Nishad/nishad_switch.py"])
