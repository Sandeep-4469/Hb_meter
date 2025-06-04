import RPi.GPIO as GPIO 
import os,cv2,csv,itertools,subprocess

def button_callback(channel):
    print("Button was pushed!")
    command=f"python3 /home/nishad/Nishad/Nishad.py"
    p=subprocess.run(command.split(" "))
    if p.returncode==0:
        print("Finished")
    
    
GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 
GPIO.add_event_detect(23,GPIO.RISING,callback=button_callback)

message = input("Press enter to quit\n\n")

GPIO.cleanup() # Clean up
