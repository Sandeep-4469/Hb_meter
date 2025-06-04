import cv2
import time

camera_index = '/dev/video3'
cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

time.sleep(2)  # warm-up

if not cap.isOpened():
    print(f"Failed to open camera at {camera_index}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:  # Check for 0 or NaN
    print("FPS not detected or zero, setting to 20")
    fps = 20

print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Data/vids/output_USB_VIS.mp4', fourcc, fps, (frame_width, frame_height))

print("Recording started...")
start_time = time.time()
while time.time() - start_time < 5:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    out.write(frame)

cap.release()
out.release()
print("Video saved as output_USB_VIS.mp4")
