#!/usr/bin/env python3
"""
RIR Dataset Visualization Tool

This script provides comprehensive visualization capabilities for the generated RIR dataset,
including time-domain plots, spectrograms, and statistical analysis of room parameters.

Usage:
    python plot_rirs.py [--dataset FILENAME] [--num_plots N] [--analysis]

Features:
    - Time-domain RIR visualization
    - Spectrogram analysis
    - Room parameter statistics
    - Reverberation time (RT60) estimation
"""

import pickle
import argparse
import sys
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from utils import get_room_parameters_from_geometry, SAMPLING_RATE, WALL_NAMES

# ---------- Configuration ----------
DEFAULT_DATASET_FILE = 'rir_dataset_walls.pkl'
DEFAULT_NUM_PLOTS = 5
FIGURE_DPI = 100
COLORMAP = 'viridis'


def load_dataset(filename: str) -> List[Dict[str, Any]]:
    """
    Load the RIR dataset from a pickle file.
    
    Args:
        filename (str): Path to the dataset pickle file
        
    Returns:
        List[Dict[str, Any]]: Loaded dataset
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        Exception: If the file cannot be loaded or is corrupted
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"Dataset file not found: {filename}")
    
    try:
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Successfully loaded dataset: {filename}")
        print(f"Dataset contains {len(dataset):,} RIR samples")
        return dataset
    except Exception as e:
        raise Exception(f"Failed to load dataset: {str(e)}")


def plot_rir_time_domain(dataset: List[Dict[str, Any]], 
                        num_plots: int = DEFAULT_NUM_PLOTS,
                        save_path: Optional[str] = None) -> None:
    """
    Plot multiple RIRs in the time domain.
    
    Args:
        dataset (List[Dict[str, Any]]): Dataset containing RIR samples
        num_plots (int): Number of RIRs to plot
        save_path (Optional[str]): Path to save the figure (if None, displays interactively)
    """
    if num_plots > len(dataset):
        num_plots = len(dataset)
        print(f"Warning: Only {len(dataset)} samples available, plotting all of them.")
    
    # Extract RIRs and convert to numpy arrays
    rirs_to_plot = [dataset[i]['rir'].numpy() for i in range(num_plots)]
    time_axis = np.arange(len(rirs_to_plot[0])) / SAMPLING_RATE
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=FIGURE_DPI)
    
    colors = plt.cm.Set1(np.linspace(0, 1, num_plots))
    
    for i, (rir, color) in enumerate(zip(rirs_to_plot, colors)):
        # Normalize RIR for better visualization
        rir_normalized = rir / np.max(np.abs(rir))
        ax.plot(time_axis, rir_normalized, label=f'RIR {i+1}', 
               color=color, linewidth=1.5, alpha=0.8)
    
    ax.set_title('Room Impulse Responses - Time Domain', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, time_axis[-1])
    
    # Add information text
    info_text = f"Sampling Rate: {SAMPLING_RATE:,} Hz\nRIR Length: {len(rirs_to_plot[0]):,} samples"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time-domain plot saved to: {save_path}")
    else:
        plt.show()


def plot_rir_spectrogram(dataset: List[Dict[str, Any]], 
                        sample_idx: int = 0,
                        save_path: Optional[str] = None) -> None:
    """
    Plot spectrogram of a single RIR.
    
    Args:
        dataset (List[Dict[str, Any]]): Dataset containing RIR samples
        sample_idx (int): Index of the RIR sample to analyze
        save_path (Optional[str]): Path to save the figure
    """
    if sample_idx >= len(dataset):
        raise IndexError(f"Sample index {sample_idx} out of range (dataset size: {len(dataset)})")
    
    rir = dataset[sample_idx]['rir'].numpy()
    
    # Create spectrogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=FIGURE_DPI)
    
    # Time domain plot
    time_axis = np.arange(len(rir)) / SAMPLING_RATE
    ax1.plot(time_axis, rir, color='blue', linewidth=1)
    ax1.set_title(f'RIR Sample {sample_idx + 1} - Time Domain', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Spectrogram
    Pxx, freqs, bins, im = ax2.specgram(rir, Fs=SAMPLING_RATE, cmap=COLORMAP)
    ax2.set_title(f'RIR Sample {sample_idx + 1} - Spectrogram', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Frequency (Hz)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Power Spectral Density (dB)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram saved to: {save_path}")
    else:
        plt.show()


def analyze_dataset_statistics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform statistical analysis of the dataset parameters.
    
    Args:
        dataset (List[Dict[str, Any]]): Dataset containing RIR samples
        
    Returns:
        Dict[str, Any]: Dictionary containing statistical analysis results
    """
    print("\n" + "="*50)
    print("DATASET STATISTICAL ANALYSIS")
    print("="*50)
    
    # Extract all geometry parameters
    all_geometries = [sample['geom'].numpy() for sample in dataset]
    
    # Room dimensions
    room_dims = np.array([geom[:3] for geom in all_geometries])
    
    # Source positions
    src_positions = np.array([geom[3:6] for geom in all_geometries])
    
    # Receiver positions  
    rec_positions = np.array([geom[6:9] for geom in all_geometries])
    
    # Absorption coefficients
    absorption_coeffs = np.array([geom[9:15] for geom in all_geometries])
    
    stats = {
        'room_dimensions': {
            'mean': np.mean(room_dims, axis=0),
            'std': np.std(room_dims, axis=0),
            'min': np.min(room_dims, axis=0),
            'max': np.max(room_dims, axis=0)
        },
        'source_positions': {
            'mean': np.mean(src_positions, axis=0),
            'std': np.std(src_positions, axis=0)
        },
        'receiver_positions': {
            'mean': np.mean(rec_positions, axis=0),
            'std': np.std(rec_positions, axis=0)
        },
        'absorption_coefficients': {
            'mean': np.mean(absorption_coeffs, axis=0),
            'std': np.std(absorption_coeffs, axis=0),
            'min': np.min(absorption_coeffs, axis=0),
            'max': np.max(absorption_coeffs, axis=0)
        }
    }
    
    # Print statistics
    print(f"Dataset size: {len(dataset):,} samples")
    print(f"RIR length: {len(dataset[0]['rir']):,} samples")
    print(f"Sampling rate: {SAMPLING_RATE:,} Hz")
    print(f"RIR duration: {len(dataset[0]['rir']) / SAMPLING_RATE:.3f} seconds")
    
    print("\nRoom Dimensions (L x W x H) [meters]:")
    print(f"  Mean: {stats['room_dimensions']['mean']}")
    print(f"  Std:  {stats['room_dimensions']['std']}")
    print(f"  Range: {stats['room_dimensions']['min']} to {stats['room_dimensions']['max']}")
    
    print(f"\nAbsorption Coefficients ({', '.join(WALL_NAMES)}):")
    print(f"  Mean: {stats['absorption_coefficients']['mean']}")
    print(f"  Std:  {stats['absorption_coefficients']['std']}")
    print(f"  Range: {stats['absorption_coefficients']['min']} to {stats['absorption_coefficients']['max']}")
    
    return stats


def plot_parameter_distributions(dataset: List[Dict[str, Any]], 
                               save_path: Optional[str] = None) -> None:
    """
    Plot distributions of room parameters.
    
    Args:
        dataset (List[Dict[str, Any]]): Dataset containing RIR samples
        save_path (Optional[str]): Path to save the figure
    """
    all_geometries = [sample['geom'].numpy() for sample in dataset]
    
    # Extract parameters
    room_dims = np.array([geom[:3] for geom in all_geometries])
    absorption_coeffs = np.array([geom[9:15] for geom in all_geometries])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=FIGURE_DPI)
    
    # Room dimension distributions
    dim_labels = ['Length (X)', 'Width (Y)', 'Height (Z)']
    for i, (ax, label) in enumerate(zip(axes[0], dim_labels)):
        ax.hist(room_dims[:, i], bins=50, alpha=0.7, color=f'C{i}', edgecolor='black')
        ax.set_title(f'Room {label} Distribution')
        ax.set_xlabel('Meters')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Absorption coefficient distributions
    for i, (ax, wall_name) in enumerate(zip(axes[1], WALL_NAMES)):
        ax.hist(absorption_coeffs[:, i], bins=50, alpha=0.7, color=f'C{i+3}', edgecolor='black')
        ax.set_title(f'{wall_name.capitalize()} Wall Absorption')
        ax.set_xlabel('Absorption Coefficient')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Dataset Parameter Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter distribution plot saved to: {save_path}")
    else:
        plt.show()


def main():
    """Main function to handle command line arguments and execute visualization."""
    parser = argparse.ArgumentParser(description='Visualize RIR dataset')
    parser.add_argument('--dataset', '-d', default=DEFAULT_DATASET_FILE,
                       help=f'Dataset pickle file (default: {DEFAULT_DATASET_FILE})')
    parser.add_argument('--num_plots', '-n', type=int, default=DEFAULT_NUM_PLOTS,
                       help=f'Number of RIRs to plot (default: {DEFAULT_NUM_PLOTS})')
    parser.add_argument('--analysis', '-a', action='store_true',
                       help='Perform statistical analysis of dataset')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save plots to files instead of displaying')
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        dataset = load_dataset(args.dataset)
        
        # Perform statistical analysis if requested
        if args.analysis:
            analyze_dataset_statistics(dataset)
            plot_parameter_distributions(dataset, 
                                       'parameter_distributions.png' if args.save else None)
        
        # Plot time-domain RIRs
        plot_rir_time_domain(dataset, args.num_plots, 
                            'rir_time_domain.png' if args.save else None)
        
        # Plot spectrogram of first RIR
        plot_rir_spectrogram(dataset, 0, 
                           'rir_spectrogram.png' if args.save else None)
        
        if args.save:
            print(f"\nAll plots saved to current directory.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
