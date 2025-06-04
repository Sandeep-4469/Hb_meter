import board
import digitalio
import time  # Import the time module
from adafruit_character_lcd.character_lcd_i2c import Character_LCD_I2C

# Define LCD parameters
lcd_columns = 16
lcd_rows = 2

# Define I2C parameters
i2c_address = 0x27  # Check your LCD's address

# Initialize I2C bus and LCD object
i2c = board.I2C()
lcd = Character_LCD_I2C(i2c, lcd_columns, lcd_rows, i2c_address)

# Print "Hello sir" on the LCD
lcd.message = "Hello sir"

# Wait for a few seconds
time.sleep(3)

# Clear the LCD
lcd.clear()

