import numpy as np
import pywt
from config import NUM_SCALES

def compute_cwt(window):
    scales = np.arange(1, NUM_SCALES + 1)
    coeffs, _ = pywt.cwt(window, scales, 'morl')
    return coeffs
