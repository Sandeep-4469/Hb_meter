from RPLCD import *
from time import sleep
from RPLCD.i2c import CharLCD

lcd = CharLCD('PCF8574', 0x27)
lcd.cursor_pos = (0, 0)
def write(text):
    lcd.write_string(text)

if __name__ == '__main__':
    write("Hello World!")
