# part5_fuzzy.py
import numpy as np

def calculate_severity(mask):
    total_pixels = mask.size
    severity = {}
    for cls in range(1, 4):  # Exclude 0 (background)
        cls_pixels = np.sum(mask == cls)
        percent = 100 * cls_pixels / total_pixels
        if percent > 50:
            label = 'Severe'
        elif percent > 20:
            label = 'Moderate'
        elif percent > 0:
            label = 'Mild'
        else:
            label = 'None'
        severity[cls] = {'percent': round(percent, 2), 'severity': label}
    return severity
