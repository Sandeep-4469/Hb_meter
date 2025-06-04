#!/usr/bin/python3

import RPi.GPIO as GPIO
import time

def led_light():
    # Disable GPIO warnings
    GPIO.setwarnings(False)

    # Define pin numbers using physical pin numbers
    PIN_RED = 33
    PIN_GREEN = 35
    PIN_BLUE = 37

    # Set up GPIO
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PIN_RED, GPIO.OUT)
    GPIO.setup(PIN_GREEN, GPIO.OUT)
    GPIO.setup(PIN_BLUE, GPIO.OUT)

    def set_color(r, g, b):
        GPIO.output(PIN_RED, r)
        GPIO.output(PIN_GREEN, g)
        GPIO.output(PIN_BLUE, b)
        print(f"Set color to R={r}, G={g}, B={b}")

    try:
        print("Setting color to Yellow")
        # Yellow: Red and Green on
        set_color(GPIO.HIGH, GPIO.HIGH, GPIO.LOW)
        time.sleep(10)

        print("Setting color to Orange")
        # Orange: Red on, Green dimmed (using PWM for better approximation)
        set_color(GPIO.HIGH, GPIO.LOW, GPIO.LOW)
        time.sleep(10)

        print("Setting color to Red")
        # Red: Only Red on
        set_color(GPIO.HIGH, GPIO.LOW, GPIO.LOW)
        time.sleep(10)
    finally:
        print("Turning off the LED")
        # Turn off all colors
        set_color(GPIO.LOW, GPIO.LOW, GPIO.LOW)
        GPIO.cleanup()

if __name__ == "__main__":
    print("Starting LED light sequence")
    led_light()
    print("LED light sequence completed")
