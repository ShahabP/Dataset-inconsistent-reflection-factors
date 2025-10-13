"""
Configuration file for RIR Dataset Generator

This module contains all configurable parameters for generating Room Impulse Response
datasets with inconsistent wall reflection factors. Modify these settings to customize
the dataset generation process.
"""

from typing import List, Tuple
import os

# =============================================================================
# DATASET GENERATION PARAMETERS
# =============================================================================

# Number of room configurations to generate
NUM_ROOMS: int = 5000

# Output settings
OUTPUT_DIR: str = "output"
DATASET_FILENAME: str = "rir_dataset_walls.pkl"
PROGRESS_INTERVAL: int = 100  # Print progress every N rooms

# =============================================================================
# ROOM IMPULSE RESPONSE PARAMETERS
# =============================================================================

# RIR characteristics
RIR_LENGTH: int = 4096              # Length of generated RIR in samples
SAMPLING_RATE: int = 16000          # Sampling rate in Hz (16 kHz is common for speech)

# Room acoustics
MAX_REFLECTION_ORDER: int = 15      # Maximum order for image source method
RAY_TRACING: bool = False          # Use ray tracing (slower but more accurate)

# =============================================================================
# ROOM GEOMETRY PARAMETERS
# =============================================================================

# Room dimension ranges (min, max) in meters
# Format: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
ROOM_DIM_RANGE: List[Tuple[float, float]] = [
    (3.0, 10.0),    # Length (X-axis): 3-10 meters
    (3.0, 10.0),    # Width (Y-axis): 3-10 meters  
    (2.5, 4.0)      # Height (Z-axis): 2.5-4 meters
]

# Minimum distance constraints to avoid numerical issues
MIN_SOURCE_WALL_DISTANCE: float = 0.1    # meters
MIN_RECEIVER_WALL_DISTANCE: float = 0.1  # meters
MIN_SOURCE_RECEIVER_DISTANCE: float = 0.5 # meters

# =============================================================================
# MATERIAL PROPERTIES
# =============================================================================

# Absorption coefficient range (0 = perfect reflection, 1 = perfect absorption)
ABSORPTION_RANGE: Tuple[float, float] = (0.2, 0.8)

# Wall names (must match pyroomacoustics convention)
WALL_NAMES: List[str] = ['east', 'west', 'north', 'south', 'ceiling', 'floor']

# Frequency-dependent materials (optional - set to True for more realistic simulation)
FREQUENCY_DEPENDENT_MATERIALS: bool = False

# Standard material presets (used if FREQUENCY_DEPENDENT_MATERIALS is True)
MATERIAL_PRESETS: dict = {
    'concrete': [0.1, 0.1, 0.2, 0.2, 0.2, 0.2],    # Low absorption
    'carpet': [0.2, 0.3, 0.6, 0.7, 0.7, 0.8],      # High absorption at high freq
    'wood': [0.15, 0.2, 0.3, 0.3, 0.25, 0.2],      # Medium absorption
    'plaster': [0.1, 0.1, 0.15, 0.2, 0.2, 0.2],    # Low-medium absorption
    'curtain': [0.3, 0.4, 0.6, 0.8, 0.8, 0.9],     # High absorption
    'glass': [0.05, 0.05, 0.1, 0.1, 0.1, 0.1],     # Very low absorption
}

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Plotting settings
DEFAULT_NUM_PLOTS: int = 5
FIGURE_DPI: int = 100
COLORMAP: str = 'viridis'
PLOT_OUTPUT_DIR: str = "plots"

# Spectrogram settings
SPECTROGRAM_NFFT: int = 1024
SPECTROGRAM_OVERLAP: int = 512

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

# Memory management
BATCH_SIZE: int = 100              # Process rooms in batches to manage memory
USE_MULTIPROCESSING: bool = False  # Enable multiprocessing (experimental)
NUM_WORKERS: int = 4               # Number of worker processes

# Data validation
VALIDATE_RIRS: bool = True         # Check RIRs for common issues
MAX_RIR_AMPLITUDE: float = 10.0    # Flag RIRs with amplitude above this threshold
MIN_RT60: float = 0.1              # Minimum expected RT60 (seconds)
MAX_RT60: float = 5.0              # Maximum expected RT60 (seconds)

# =============================================================================
# FILE I/O PARAMETERS
# =============================================================================

# Pickle protocol for saving datasets
PICKLE_PROTOCOL: int = 4  # Use protocol 4 for compatibility with Python 3.4+

# Compression settings
COMPRESS_OUTPUT: bool = False      # Compress output files (slower but smaller)
COMPRESSION_LEVEL: int = 6         # Compression level (1-9, higher = more compression)

# =============================================================================
# LOGGING AND DEBUG PARAMETERS
# =============================================================================

# Logging configuration
LOG_LEVEL: str = "INFO"            # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE: bool = False          # Save logs to file
LOG_FILENAME: str = "rir_generation.log"

# Debug settings
DEBUG_MODE: bool = False           # Enable detailed debug output
SAVE_INTERMEDIATE_RESULTS: bool = False  # Save intermediate computation results
PLOT_DEBUG_ROOMS: bool = False     # Generate debug plots for each room

# =============================================================================
# RANDOM SEED SETTINGS
# =============================================================================

# Reproducibility settings
USE_RANDOM_SEED: bool = False      # Set to True for reproducible results
RANDOM_SEED: int = 42              # Random seed value

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_output_path(filename: str) -> str:
    """Get full path for output file, creating directory if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)

def get_plot_path(filename: str) -> str:
    """Get full path for plot file, creating directory if needed."""
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    return os.path.join(PLOT_OUTPUT_DIR, filename)

def validate_config() -> bool:
    """Validate configuration parameters for consistency."""
    errors = []
    
    # Check room dimensions
    for i, (min_dim, max_dim) in enumerate(ROOM_DIM_RANGE):
        if min_dim >= max_dim:
            errors.append(f"Room dimension {i}: min ({min_dim}) >= max ({max_dim})")
        if min_dim <= 0:
            errors.append(f"Room dimension {i}: min ({min_dim}) <= 0")
    
    # Check absorption range
    if ABSORPTION_RANGE[0] >= ABSORPTION_RANGE[1]:
        errors.append(f"Absorption range: min ({ABSORPTION_RANGE[0]}) >= max ({ABSORPTION_RANGE[1]})")
    if not (0 <= ABSORPTION_RANGE[0] <= 1 and 0 <= ABSORPTION_RANGE[1] <= 1):
        errors.append(f"Absorption coefficients must be in [0, 1], got {ABSORPTION_RANGE}")
    
    # Check other parameters
    if RIR_LENGTH <= 0:
        errors.append(f"RIR_LENGTH must be positive, got {RIR_LENGTH}")
    if SAMPLING_RATE <= 0:
        errors.append(f"SAMPLING_RATE must be positive, got {SAMPLING_RATE}")
    if NUM_ROOMS <= 0:
        errors.append(f"NUM_ROOMS must be positive, got {NUM_ROOMS}")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

# Validate configuration on import
if __name__ == "__main__":
    if validate_config():
        print("Configuration validation: PASSED")
    else:
        print("Configuration validation: FAILED")