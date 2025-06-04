import subprocess

def start(output_file, timer=5, fps=30):
    def record_video():
        command = f"ffmpeg -f v4l2 -video_size 1920x1080 -input_format mjpeg -i /dev/video3 -c:v copy -t {timer} -r {fps} {output_file}"
        status_output = subprocess.getstatusoutput(command)
        print(status_output)
        if status_output[0] == 0:
            print("Recording completed successfully.")
        else:
            print("Error during recording:", status_output[1])

    # Start recording the video
    record_video()

# Example usage
start('/home/nishad/output.mp4')
