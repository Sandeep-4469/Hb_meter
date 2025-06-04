import RPi.GPIO as GPIO
import subprocess
import time
from rpi_lcd import LCD
from signal import signal, SIGTERM, SIGHUP

# Define GPIO pin for the switch
SWITCH_PIN = 16

# Initialize LCD
lcd = LCD()

# Set up GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Define safe exit function for LCD
def safe_exit(signum, frame):
    lcd.clear()
    GPIO.cleanup()
    exit(1)

signal(SIGTERM, safe_exit)
signal(SIGHUP, safe_exit)

# Function to display ready message for 5 seconds
def display_ready_message():
    lcd.clear()
    lcd.text("Press button to", 1)
    lcd.text("take reading", 2)
    time.sleep(5)
    lcd.clear()

# Main loop
try:
    # Display ready message initially
    display_ready_message()
    
    while True:
        # Check if the button is pressed
        if GPIO.input(SWITCH_PIN) == GPIO.HIGH:
            # Display "Executing script..." message
            lcd.clear()
            lcd.text("Executing script...", 1)
            print("Button was pushed! Executing Nishad.py...")  # Debug statement

            try:
                # Run the script
                subprocess.run(["python3", "/home/nishad/Nishad/Nishad.py"])
            except Exception as e:
                print(f"Error occurred while running Nishad.py: {e}")

            # Wait 10 seconds before allowing another press
            time.sleep(10)

            # Display ready message again after 10 seconds
            display_ready_message()

        # Brief delay to avoid button bouncing issues
        time.sleep(0.1)

except KeyboardInterrupt:
    pass

finally:
    GPIO.cleanup()
    lcd.clear()
