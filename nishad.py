import os
import time
import subprocess
from signal import signal, SIGTERM, SIGHUP
from rpi_lcd import LCD

# Import specific functions from your AllinOne.py script
from AllinOne import (
    unique_file,
    start_dual,
    # start_single, # Uncomment if you use single camera
    extract_features_for_video_angle,
    load_pytorch_model,
    predict_pytorch,
    display_on_lcd # The display for the final result
)

# Ensure the working directory is set correctly for relative paths
os.chdir("/home/nishad/Nishad/")

# --- Define paths for recording and feature output ---
VIDEO_RECORDING_DIR = "Data/vids/"
FINAL_HISTOGRAM_DIR = "Data/final_hist_for_inference/" # Directory for the (256x9) histogram output
MODEL_PATH_PYTORCH = "/home/nishad/Nishad/best_model_99.pth" # Path to your PyTorch model file

# List of directories to ensure exist
paths_to_create = [VIDEO_RECORDING_DIR, FINAL_HISTOGRAM_DIR]

# --- Parameters for Recording and Feature Extraction ---
VIDEO_BASE_NAME = "record" # Base name for recorded video files
RECORDING_TIMER = 30       # Duration of video recording in seconds (was 30s in original main)
FPS = 30                   # Frames per second for recording

# Frame range for feature extraction (must match training)
# Based on your training script: frames 50 to 590 inclusive (total 541 frames)
START_FRAME_EXTRACTION = 50
END_FRAME_EXTRACTION_EXCLUSIVE = 591 # Processes frames up to END_FRAME_EXTRACTION_EXCLUSIVE - 1

# --- LCD Display Helper Functions (Specific to this main workflow) ---
# These are used for user prompts like "Lets Start", "Please wait", etc.
def display_message_on_lcd(line1, line2, duration):
    lcd = LCD()
    try:
        lcd.clear()
        lcd.text(line1, 1)
        lcd.text(line2, 2)
        time.sleep(duration)
    except KeyboardInterrupt:
        print("LCD message display interrupted.")
    except Exception as e:
        print(f"LCD error in display_message_on_lcd: {e}")
    finally:
        lcd.clear()

def display_greet():
    display_message_on_lcd("Lets Start", "the procedure", 5)

def display_waiting_for_recording():
    # Ensure the message matches the actual recording time set by RECORDING_TIMER
    display_message_on_lcd("Please dont move,", f"for {RECORDING_TIMER} seconds", 5) # Display initial message, not full duration

def display_recording_successful():
    display_message_on_lcd("Reading complete", "Remove finger", 5)

def display_computation_ongoing():
    display_message_on_lcd("Computation ", "Ongoing", 5)

def display_for_new_reading():
    display_message_on_lcd("For new reading", "Wait 10 second", 10)

# --- Main Script Execution ---
if __name__ == "__main__":
    # Create directories if they don't exist
    for p_dir in paths_to_create:
        if not os.path.exists(p_dir):
            os.makedirs(p_dir)
            print(f"Created directory: {p_dir}")

    # Check if recording timer is sufficient for feature extraction range
    min_frames_needed = END_FRAME_EXTRACTION_EXCLUSIVE
    min_duration_needed_s = min_frames_needed / FPS
    if RECORDING_TIMER < min_duration_needed_s:
        print(f"Warning: RECORDING_TIMER ({RECORDING_TIMER}s) may be too short for "
              f"END_FRAME_EXTRACTION_EXCLUSIVE ({END_FRAME_EXTRACTION_EXCLUSIVE} at {FPS}fps requires {min_duration_needed_s:.2f}s).")
        print("Adjusting RECORDING_TIMER to ensure sufficient frames are captured.")
        # Automatically adjust the recording timer if it's too short
        RECORDING_TIMER = int(min_duration_needed_s + 2) # Add 2 seconds buffer
        print(f"RECORDING_TIMER set to: {RECORDING_TIMER}s")


    try:
        # 1. Generate unique filename for the video capture
        unique_name_part, actual_video_file_path = unique_file(os.path.join(VIDEO_RECORDING_DIR, VIDEO_BASE_NAME))

        # 2. Determine the path for the final histogram CSV output
        # This is where extract_features_for_video_angle will save the 256x9 histogram
        final_histogram_for_model_path = os.path.join(FINAL_HISTOGRAM_DIR, f"{unique_name_part}_avg_hist.csv")

        print(f"Video file to be created: {actual_video_file_path}")
        print(f"Final Average Histogram CSV to be created: {final_histogram_for_model_path}")

        # --- Start Workflow ---
        display_greet()

        display_waiting_for_recording()

        # 3. Record video (and turn on LED)
        # Use start_dual or start_single based on your camera setup
        # start_dual(actual_video_file_path, timer=RECORDING_TIMER, fps=FPS)
        start_single(actual_video_file_path, timer=RECORDING_TIMER, fps=FPS) # Example using single camera

        display_recording_successful()

        # Check if video was created successfully
        if not os.path.exists(actual_video_file_path) or os.path.getsize(actual_video_file_path) == 0:
            print(f"Critical Error: Video recording failed or produced an empty file: {actual_video_file_path}")
            display_message_on_lcd("Video Error", "Rec failed", 10)
            # Decide if you want to exit or try cleanup
            # For a critical failure like no video, exiting is reasonable
            exit(1)

        # 4. Extract features (average histogram) from the recorded video
        # This function performs the per-pixel analysis and averages frame histograms
        display_computation_ongoing() # Inform user computation is starting
        print("Starting feature extraction process...")

        feature_extraction_success = extract_features_for_video_angle(
            video_path_or_capture=actual_video_file_path,
            output_hist_csv_path=final_histogram_for_model_path,
            angle=0, # Use angle 0 for inference (no rotation)
            start_frame=START_FRAME_EXTRACTION,
            end_frame_exclusive=END_FRAME_EXTRACTION_EXCLUSIVE
        )

        if not feature_extraction_success:
            print("Critical Error: Feature extraction failed.")
            display_message_on_lcd("Feature Error", "Extraction fail", 10)
            exit(1) # Stop execution if feature extraction failed

        # 5. Load the trained PyTorch Model
        print("Loading PyTorch model...")
        # Determine device (CPU or GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        pytorch_model = load_pytorch_model(MODEL_PATH_PYTORCH, device)

        if pytorch_model is None:
            print("Critical Error: Failed to load PyTorch model.")
            display_message_on_lcd("Model Error", "Load failed", 10)
            exit(1) # Stop execution if model failed to load

        # 6. Predict Hb value using the PyTorch model
        # The model input is the final_histogram_for_model_path
        print(f"Predicting using features from: {final_histogram_for_model_path}")
        hb_prediction_value, anemia_status = predict_pytorch(
            pytorch_model,
            final_histogram_for_model_path, # Pass the path to the (256x9) histogram CSV
            device
        )

        # 7. Display the final result on the LCD
        if hb_prediction_value is None:
            print("Error during prediction.")
            display_message_on_lcd("Prediction Err", "Check logs", 10)
        else:
            print(f"Predicted Hb: {hb_prediction_value:.2f}, Status: {anemia_status}")
            # display_on_lcd comes from AllinOne.py and shows the actual result
            display_on_lcd(hb_prediction_value, anemia_status)

        # 8. Display message instructing for the next reading
        display_for_new_reading()

        print("Main script finished one cycle.")

        # Optional: Rerun another script (e.g., a loop controller)
        # subprocess.run(["python3", "/home/nishad/Nishad/nishad_switch.py"])

    except Exception as e:
        # Catch any unexpected errors in the main execution flow
        print(f"An unexpected error occurred in the main execution block: {e}")
        import traceback
        traceback.print_exc() # Print detailed error info

        try:
            display_message_on_lcd("System Error", "Check logs", 10)
        except Exception as e_lcd:
            print(f"Additionally, failed to display error message on LCD: {e_lcd}")

    finally:
        # Ensure GPIO pins are cleaned up regardless of success or failure
        print("Script finished. Cleaning up GPIO.")
        GPIO.cleanup()