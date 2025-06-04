import RPi.GPIO as GPIO
import time

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
