from scipy.signal import find_peaks
import numpy as np

def find_gamma_peaks(t, y, height_threshold=0.1):
    """
    Find peaks in the dual tracer signal and set the highest peak to tracer 1 and the second highest to tracer 2.
    Args: dual_tracer_signal: 1D array of the dual tracer signal
          height_threshold: minimum height of peaks to consider
          Returns: peak_1, peak_2: indices of the peaks corresponding to tracer 1 and tracer 2
          """
    peaks, properties = find_peaks(y, height=height_threshold)
    if len(peaks) < 2:
        raise ValueError("Not enough peaks found in the signal to separate tracers.")
    
    # Sort peaks by height
    sorted_indices = np.argsort(properties["peak_heights"])[::-1]
    peak_1 = peaks[sorted_indices[0]]  # Highest peak for tracer 1
    peak_2 = peaks[sorted_indices[1]]  # Second highest peak for tracer 2
    
    return peak_1, peak_2
    