#!/bin/bash
#echo "Startup script executed at: $(date)" >> /home/pi/startup.log

# Activate virtual env
source /home/nishad/Nishad_env/bin/activate

# Run the python program
python3 /home/nishad/Nishad/nishad_switch.py