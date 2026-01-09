# Non-Invasive Hemoglobin (Hb) Estimation System

## Overview

This project implements a **non-invasive hemoglobin (Hb) estimation framework** using fingertip videos captured under **controlled multi-wavelength LED illumination**. The system extracts physiological signals from video data and applies machine intelligence techniques to estimate hemoglobin concentration without invasive blood sampling.

The solution was validated against **clinical hemoglobin measurements** and translated from a research prototype into a **real-world embedded deployment**, with a focus on robustness, computational efficiency, and system-level constraints.

---

## Key Features

- Non-invasive hemoglobin estimation using fingertip videos  
- Multi-wavelength LED–based illumination control  
- Video-based PPG signal extraction  
- Machine learning regression for Hb prediction  
- Validation against clinical reference measurements  
- Embedded deployment optimized for real-time performance  

---

## System Pipeline

1. **Data Acquisition**
   - Fingertip videos captured under controlled multi-wavelength LED illumination
   - Camera-based sensing

2. **Preprocessing**
   - Region of Interest (ROI) selection
   - Noise reduction and signal normalization

3. **Feature Extraction**
   - Converted Video into histograms and is converted into csv files, one for each respectively

4. **Modeling**
   - These csv files are converted into tensors and feeded into trained Model consisting of 3 heads. one for each csv

5. **Deployment**
   - Optimized inference pipeline for embedded hardware
   - Low-latency and resource-efficient implementation

---

## Custom Evaluation Metric

A custom metric is used to jointly optimize **average accuracy** and **worst-case robustness** by combining **Mean Squared Error (MSE)** and **Maximum Error (ME)**.

### Metric Definition

CustomMetric = α · MSE + (1 − α) · ME

where:
- **MSE**: Mean Squared Error  
- **ME**: Maximum Absolute Error  
- **α ∈ [0, 1]**: weighting factor controlling the trade-off  

---

### Mathematical Formulation

Let \( y_i \) be the clinical hemoglobin values and \( \hat{y}_i \) be the predicted values:

- MSE = (1 / N) · Σ (yᵢ − ŷᵢ)²  
- ME = max |yᵢ − ŷᵢ|

Final metric:

α · (1 / N) · Σ (yᵢ − ŷᵢ)² + (1 − α) · max |yᵢ − ŷᵢ|

---

### Python Implementation

```python
import numpy as np

def custom_hb_metric(y_true, y_pred, alpha=0.7):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    max_error = np.max(np.abs(y_true - y_pred))

    return alpha * mse + (1 - alpha) * max_error


