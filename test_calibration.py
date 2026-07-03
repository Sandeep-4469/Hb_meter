import os
import json
import joblib
from calibration_utils import calibrate_prediction

def test_calibration():
    # 1. Create a dummy patient info with clinical data (Severe Anemia)
    patient_anemic = {
        "Age": 23.0,
        "Gender": "F",
        "RBC": 3.1,
        "MCV": 61.1,
        "MCH": 18.6
    }
    
    # Create a dummy patient info with clinical data (Normal Male)
    patient_normal = {
        "Age": 20.0,
        "Gender": "M",
        "RBC": 5.66,
        "MCV": 84.2,
        "MCH": 28.3
    }
    
    print("Testing Calibration Models...")
    print("-" * 50)
    
    # 2. Test Clinical Calibration
    # Anemic patient: actual Hb is ~5.8. Raw prediction from unpatched meter was 11.34.
    pred_anemic_cl = calibrate_prediction(11.34, patient_anemic)
    print(f"Clinical Calibrated (Anemic): {pred_anemic_cl:.2f} g/dL (Expected: ~5.8)")
    
    # Normal patient: actual Hb is ~16.0. Raw prediction from unpatched meter was 11.64.
    pred_normal_cl = calibrate_prediction(11.64, patient_normal)
    print(f"Clinical Calibrated (Normal): {pred_normal_cl:.2f} g/dL (Expected: ~16.0)")
    
    # 3. Test Non-Invasive Calibration
    pred_anemic_ni = calibrate_prediction(11.34, {"Age": 23.0, "Gender": "F"})
    print(f"Non-Invasive Calibrated (Anemic): {pred_anemic_ni:.2f} g/dL")
    
    pred_normal_ni = calibrate_prediction(11.64, {"Age": 20.0, "Gender": "M"})
    print(f"Non-Invasive Calibrated (Normal): {pred_normal_ni:.2f} g/dL")
    
    print("-" * 50)
    print("Test complete!")

if __name__ == "__main__":
    test_calibration()
