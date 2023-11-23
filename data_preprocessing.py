# Load necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

# Load and preprocess Raman spectra data
def preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)  # Replace with your actual file format and path
    
    # Normalize spectra
    spectra = data.iloc[:, 1:]  # Assuming the spectral data starts from column 1
    normalized_spectra = normalize(spectra, axis=1, norm='l1')  # Normalize spectra
    
    # Perform baseline correction and noise reduction as needed

    return normalized_spectra
