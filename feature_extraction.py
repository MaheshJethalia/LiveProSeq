# Implement peak detection and intensity measurement
# Example: Using libraries like scipy for peak detection
from scipy.signal import find_peaks

def extract_features(spectra):
    # Peak detection
    peaks, _ = find_peaks(spectra[0])  # Example: Using the first spectrum for peaks
    
    # Intensity measurement
    peak_intensities = spectra[:, peaks]  # Get intensities at detected peaks
    
    return peaks, peak_intensities
