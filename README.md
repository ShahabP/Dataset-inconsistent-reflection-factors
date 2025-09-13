# Room Impulse Response (RIR) Dataset Generator

This repository provides a framework to generate synthetic **Room Impulse Responses (RIRs)** for shoebox-shaped rooms.  
Each room is simulated with **different and inconsistent wall reflection coefficients**, producing realistic variations in reverberation across surfaces. The dataset is ideal for research and development in **speech enhancement, dereverberation, source localization, and acoustic modeling**.

## Features

- Generate RIRs for **5000 unique room setups** with variable dimensions, source-receiver positions, and wall properties.
- Wall reflection factors are **different and inconsistent across all walls** in each room, simulating realistic acoustic variability.
- Frequency-dependent absorption profiles supported for more realistic simulations.
- Built-in scripts to **visualize and plot example RIRs**.
- Flexible configuration to control room sizes, reverberation times, and sampling rates.

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository_url>
cd rir_dataset_generator
pip install -r requirements.txt
