# Room Impulse Response (RIR) Dataset Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About

This repository provides a framework for generating a large-scale dataset of Room Impulse Responses (RIRs). Specifically, it can simulate 5000 unique room configurations, each with distinct geometry and microphone–source placement. Within each shoebox room, all walls may be assigned different and inconsistent reflection/absorption coefficients to model irregular wall properties. In addition, the repository includes a simple generator for L-shaped (non-convex) rooms (a rectangular room with a rectangular cutout) to synthesize RIRs for concave geometries.

Both dataset types are intended for acoustic research and machine learning experiments such as dereverberation, source localization, and acoustic scene analysis.

## 🎯 Overview

This repository generates synthetic RIRs for shoebox-shaped rooms where each wall has **different and inconsistent reflection coefficients**, creating realistic acoustic environments. The generated dataset is particularly valuable for research in:

- **Speech Enhancement** and Dereverberation
- **Source Localization** and Separation
- **Acoustic Scene Analysis**
- **Machine Learning** model training for acoustic tasks
- **Room Acoustics** simulation and modeling

## ✨ Key Features

- **Large-Scale Generation**: Create up to 5000+ unique room configurations
- **Realistic Variability**: Each wall has independent, randomized absorption coefficients
- **Flexible Parameters**: Configurable room dimensions, source/receiver positions, and acoustic properties
- **Multiple Formats**: Output in both NumPy and PyTorch tensor formats
- **Visualization Tools**: Built-in plotting utilities for RIR analysis
- **Research-Ready**: Structured data format suitable for ML training pipelines
- **L-Shaped Room Generation**: Simple generator to synthesize RIRs for concave L-shaped rooms

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShahabP/Dataset-inconsistent-reflection-factors.git
   cd Dataset-inconsistent-reflection-factors
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Generating the Dataset

Generate a dataset of 5000 RIRs with inconsistent wall reflection factors:

```bash
python generate_rir_dataset_walls.py
```

### L-Shaped (Non-Convex) Rooms

This repository now includes a simple generator for L-shaped (concave) rooms.
The model is intentionally simple: it builds a planar L-shaped polygon (a
rectangular room with a rectangular cutout) and uses pyroomacoustics' polygon
room support (extruded to 3D when available) to synthesize RIRs.

Generate 5000 L-shaped RIRs (default):

```bash
python generate_lshaped_rirs.py
```

The output file is `output/rir_dataset_l_shaped.pkl` and contains tuples of
`(geometry_vector, rir)` where `geometry_vector` stores `[Lx, Ly, cut_w, cut_h, height, src_xyz, rec_xyz, absorption]`.

This will create `rir_dataset_walls.pkl` containing:
- **Geometry vectors**: Room dimensions, source/receiver positions, and wall absorption coefficients
- **RIR data**: 4096-sample impulse responses at 16 kHz sampling rate

### Visualizing Results

Plot sample RIRs from the generated dataset:

```bash
python plot_rirs.py
```

### Programmatic Usage

```python
from utils import create_rir
import pickle

# Generate a single RIR
geom_vector, rir = create_rir()

# Load existing dataset
with open('rir_dataset_walls.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Access data
geometry = dataset[0]['geom']  # [room_dims(3) + src_pos(3) + rec_pos(3) + abs_coeffs(6)]
impulse_response = dataset[0]['rir']  # 4096-length RIR
```

## 🏗️ Technical Specifications

### Room Configuration
- **Dimensions**: 3-10m (length/width), 2.5-4m (height)
- **Shapes supported**: Shoebox rooms with per-wall, independent absorption coefficients (irregular walls); and simple L-shaped (concave) rooms via the L-shaped generator.
- **Absorption Range**: 0.2-0.8 per wall surface (shoebox). The L-shaped generator currently uses a simple single absorption value for the whole room model.
- **Wall Surfaces**: 6 independent surfaces (east, west, north, south, ceiling, floor) for shoebox rooms

### RIR Properties
- **Sampling Rate**: 16 kHz
- **Length**: 4096 samples (~0.256 seconds)
- **Method**: Image source method with up to 15th order reflections
- **Output Format**: PyTorch tensors (Float32)

### Dataset Structure
The repository produces two dataset variants with slightly different geometry encodings:

- **Shoebox rooms** (`rir_dataset_walls.pkl`):
   - **Geometry Vector** (15 elements):
      - Room dimensions: `[Lx, Ly, Lz]`
      - Source position: `[src_x, src_y, src_z]`
      - Receiver position: `[rec_x, rec_y, rec_z]`
      - Wall absorption coefficients: `[east, west, north, south, ceiling, floor]`

- **L-shaped rooms** (`rir_dataset_l_shaped.pkl`):
   - **Geometry Vector** (12 elements):
      - `[Lx, Ly, cut_w, cut_h, height, src_x, src_y, src_z, rec_x, rec_y, rec_z, absorption]`
      - Note: the L-shaped generator uses a single absorption value for simplicity; expandable to per-wall values if desired.

- **RIR**: 4096-sample impulse response (both dataset types)

## 📁 Repository Structure

```
Dataset-inconsistent-reflection-factors/
├── generate_rir_dataset_walls.py   # Main dataset generation script
├── generate_lshaped_rirs.py        # New: L-shaped room RIR generator
├── plot_rirs.py                    # Visualization utility
├── utils.py                        # Core RIR generation functions
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔬 Research Applications

This dataset is designed for:

1. **Machine Learning Training**: 
   - Neural network training for dereverberation
   - Acoustic parameter estimation
   - Room classification tasks

2. **Algorithm Development**:
   - Speech enhancement algorithm testing
   - Reverberation modeling research
   - Source localization benchmarking

3. **Acoustic Analysis**:
   - Room acoustics simulation validation
   - Reverberation time analysis
   - Absorption coefficient studies

## ⚙️ Configuration

Key parameters can be modified in `utils.py`:

```python
# RIR generation parameters
rir_length = 4096              # RIR length in samples
fs = 16000                     # Sampling rate (Hz)
room_dim_range = [(3,10), (3,10), (2.5,4)]  # [x, y, z] room dimensions (m)
absorption_range = [0.2, 0.8]  # Min/max absorption coefficients
```

## 📋 Requirements

- `numpy`: Numerical computing
- `torch`: PyTorch tensors and operations
- `pyroomacoustics`: Room acoustics simulation
- `matplotlib`: Plotting and visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Submit feature requests
- Create pull requests with improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🔗 References

- [Pyroomacoustics Documentation](https://pyroomacoustics.readthedocs.io/)
