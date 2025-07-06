import RPi.GPIO as GPIO
import subprocess
import time
from rpi_lcd import LCD
from signal import signal, SIGTERM, SIGHUP
import sys
import threading
import traceback
import os

# Constants
SWITCH_PIN = 23  # BCM GPIO pin number
LOG_FILE = "/home/nishad/Nishad/nishad_log.txt"
SCRIPT_PATH = "/home/nishad/Nishad/Nishad.py"

lcd = LCD()
exit_flag = threading.Event()

# Logging helper
def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.ctime()}] {msg}\n")

# Clean exit handler
def safe_exit(signum=None, frame=None):
    try:
        exit_flag.set()
        GPIO.remove_event_detect(SWITCH_PIN)
        lcd.clear()
        GPIO.cleanup()
        log("Cleaned up GPIO and LCD.")
    except Exception as e:
        log(f"Cleanup error: {e}")
    sys.exit(0)

# Display initial message
def display_ready_message():
    lcd.clear()
    lcd.text("Press button to", 1)
    lcd.text("take reading", 2)

# Button press handler
def handle_button_press(channel):
    if exit_flag.is_set():
        return
    lcd.clear()
    lcd.text("Executing script...", 1)
    log("Button pressed. Running Nishad.py")

    try:
        subprocess.run(["python3", SCRIPT_PATH])
    except Exception as e:
        error_msg = f"Error: {e}"
        log(error_msg)
        lcd.text("Script error", 1)
        lcd.text(str(e)[:16], 2)  # Truncate error
        time.sleep(3)

    display_ready_message()

def button_callback():
    print("Button was pushed!")
    command=f"python3 /home/nishad/Nishad/Nishad.py"
    p=subprocess.run(command.split(" "))
    if p.returncode==0:
        print("Finished")


# Main function
def main():
    SWITCH_PIN = 23
    try:
        # log("Script starting...")
        # time.sleep(5)  # Allow system boot time

        # log(f"/dev/gpiomem exists? {os.path.exists('/dev/gpiomem')}")

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down = GPIO.PUD_UP)

        # GPIO.setmode(GPIO.BCM)
        # GPIO.cleanup()  # Important to clear old settings
        # GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        while True:
            print(SWITCH_PIN)
            if GPIO.input(SWITCH_PIN)==GPIO.LOW:
                display_ready_message()
                button_callback()
                print("something")
                # break
            else:
                print("anything")

            time.sleep(0.2)

        # display_ready_message()

        # GPIO.add_event_detect(SWITCH_PIN, GPIO.RISING, callback=handle_button_press, bouncetime=300)
        # log("GPIO event detection added.")

        while not exit_flag.is_set():
            time.sleep(1)

    except KeyboardInterrupt:
        log("KeyboardInterrupt received.")
        safe_exit()

    except RuntimeError as e:
        log(f"RuntimeError: {e}")
        lcd.clear()
        lcd.text("GPIO error", 1)
        lcd.text(str(e)[:16], 2)
        time.sleep(5)
        safe_exit()

    except Exception as e:
        log("Fatal exception:\n" + traceback.format_exc())
        lcd.clear()
        lcd.text("Fatal error", 1)
        time.sleep(5)
        safe_exit()

if __name__ == "__main__":
    signal(SIGTERM, safe_exit)
    signal(SIGHUP, safe_exit)
    main()
