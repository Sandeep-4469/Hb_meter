import RPi.GPIO as GPIO
from time import sleep

LED_R_PIN = 26
LED_G_PIN = 19
LED_B_PIN = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup([LED_R_PIN, LED_G_PIN, LED_B_PIN], GPIO.OUT)

def set_color_intensity(red_intensity, green_intensity, blue_intensity):
    GPIO.output(LED_R_PIN, GPIO.HIGH if red_intensity > 0 else GPIO.LOW)
    GPIO.output(LED_G_PIN, GPIO.HIGH if green_intensity > 0 else GPIO.LOW)
    GPIO.output(LED_B_PIN, GPIO.HIGH if blue_intensity > 0 else GPIO.LOW)

    # Adjust the sleep time based on the maximum intensity value
    max_intensity = max(red_intensity, green_intensity, blue_intensity)
    sleep_time = max(1, max_intensity / 255 * 10)  # Ensure minimum sleep of 1 second

    sleep(sleep_time)

try:
    while True:
        # Turn on Yellow (Red + Green) for 10 seconds
        print("Turning on Yellow for 10 seconds")
        set_color_intensity(255, 255, 0)
        
        # Turn on Orange (Red + some Green) for 10 seconds
        print("Turning on Orange for 10 seconds")
        set_color_intensity(255, 150, 0)
        
        # Turn on Red (Only Red) for 10 seconds
        print("Turning on Red for 10 seconds")
        set_color_intensity(255, 0, 0)

except KeyboardInterrupt:
    print("Stopping program by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Turn off all LEDs
    GPIO.output(LED_R_PIN, GPIO.LOW)
    GPIO.output(LED_G_PIN, GPIO.LOW)
    GPIO.output(LED_B_PIN, GPIO.LOW)
    # Cleanup GPIO settings
    GPIO.cleanup()
