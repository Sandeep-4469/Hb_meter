import cv2
import time
def record_video(output_file, duration=30, fps=25):
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (640, 480))
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Write the frame to the video file
        out.write(frame)
        
        # Check if the duration has been reached
        if time.time() - start_time > duration:
            break
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("Recording complete.")

# Example usage
record_video('video.mp4', duration=30)
