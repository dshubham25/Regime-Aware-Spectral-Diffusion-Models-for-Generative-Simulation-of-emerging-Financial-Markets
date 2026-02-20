import numpy as np
from config import WINDOW_TOTAL, STEP_SIZE

def create_windows(series):
    windows = []
    for i in range(0, len(series) - WINDOW_TOTAL, STEP_SIZE):
        window = series[i:i + WINDOW_TOTAL]
        windows.append(window)
    return np.array(windows)
