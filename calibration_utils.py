import os
import numpy as np
import joblib

def calibrate_prediction(raw_pred, patient_info=None):
    """
    Calibrates the raw hemoglobin prediction using the trained calibration models.
    
    Parameters:
        raw_pred (float): The raw Hb prediction from the PyTorch model.
        patient_info (dict): Optional dict containing patient clinical/demographic features:
            - 'Age' (float): Patient age. Default is 28.0.
            - 'Gender' (str): 'F' or 'M'. Default is 'F'.
            - 'RBC' (float): Red Blood Cell count (million/uL). Optional.
            - 'MCV' (float): Mean Corpuscular Volume (fL). Optional.
            - 'MCH' (float): Mean Corpuscular Hemoglobin (pg). Optional.
            
    Returns:
        float: Calibrated hemoglobin prediction (g/dL).
    """
    if patient_info is None:
        patient_info = {}
        
    # Get demographic values with sensible defaults (medians from hospital data)
    age = float(patient_info.get('Age', 28.0))
    gender_str = str(patient_info.get('Gender', 'F')).strip().upper()
    gender_bin = 1.0 if gender_str == 'M' else 0.0
    
    # Check if we have complete clinical data for clinical calibration
    rbc = patient_info.get('RBC')
    mcv = patient_info.get('MCV')
    mch = patient_info.get('MCH')
    
    # Check if models exist
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_cl_path = os.path.join(dir_path, 'calibration_clinical.joblib')
    model_ni_path = os.path.join(dir_path, 'calibration_non_invasive.joblib')
    
    # Mode A: Clinical Calibration (requires RBC, MCV, MCH)
    if rbc is not None and mcv is not None and mch is not None:
        if os.path.exists(model_cl_path):
            try:
                model = joblib.load(model_cl_path)
                features = np.array([[raw_pred, age, gender_bin, float(rbc), float(mcv), float(mch)]])
                cal_pred = float(model.predict(features)[0])
                print(f"Applying clinical calibration. Inputs: raw={raw_pred:.2f}, age={age}, gender={gender_str}, RBC={rbc}, MCV={mcv}, MCH={mch} -> calibrated={cal_pred:.2f}")
                return cal_pred
            except Exception as e:
                print(f"Error in clinical calibration: {e}. Falling back to non-invasive.")
        else:
            print("Clinical calibration model file not found. Falling back to non-invasive.")
            
    # Mode B: Non-Invasive Calibration (only age/gender)
    if os.path.exists(model_ni_path):
        try:
            model = joblib.load(model_ni_path)
            features = np.array([[raw_pred, age, gender_bin]])
            cal_pred = float(model.predict(features)[0])
            print(f"Applying non-invasive calibration. Inputs: raw={raw_pred:.2f}, age={age}, gender={gender_str} -> calibrated={cal_pred:.2f}")
            return cal_pred
        except Exception as e:
            print(f"Error in non-invasive calibration: {e}. Returning raw.")
    else:
        print("Non-invasive calibration model file not found. Returning raw.")
        
    return raw_pred
