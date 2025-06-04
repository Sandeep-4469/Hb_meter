#!/bin/bash

# Get packages required for OpenCV

#sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
#sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
#sudo apt-get -y install libxvidcore-dev libx264-dev
#sudo apt-get -y install qt4-dev-tools 
#sudo apt-get -y install libatlas-base-dev

# Need to get an older version of OpenCV because version 4 has errors
c
pip3 install pandas==1.5.3
pip3 install opencv-python==4.7.0.72
python3 -m pip install tflite-runtime

# Get packages required for TensorFlow
# Using the tflite_runtime packages available at https://www.tensorflow.org/lite/guide/python
# Will change to just 'pip3 install tensorflow' once newer versions of TF are added to piwheels

#pip3 install tensorflow

