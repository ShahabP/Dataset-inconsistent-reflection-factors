#!/usr/bin/env python3
"""
Example usage script for the RIR Dataset Generator

This script demonstrates various ways to use the RIR dataset generation tools,
including basic usage, custom configurations, and analysis workflows.
"""

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from utils import create_rir, get_room_parameters_from_geometry
from config import (
    SAMPLING_RATE, DATASET_FILENAME, ABSORPTION_RANGE, 
    get_output_path, get_plot_path
)
from generate_lshaped_rirs import generate_dataset as generate_l_shaped_dataset

def example_1_basic_usage():
    """Example 1: Basic RIR generation and inspection."""
    print("=" * 50)
    print("EXAMPLE 1: Basic RIR Generation")
    print("=" * 50)
    
    # Generate a single RIR
    print("Generating a single RIR...")
    geometry_vector, rir = create_rir()
    
    # Extract parameters
    params = get_room_parameters_from_geometry(geometry_vector)
    
    # Print information
    print(f"\nRoom Parameters:")
    print(f"  Dimensions: {params['room_dimensions']} meters")
    print(f"  Source position: {params['source_position']} meters")
    print(f"  Receiver position: {params['receiver_position']} meters")
    print(f"\nWall Absorption Coefficients:")
    for wall, coeff in params['absorption_coefficients'].items():
        print(f"  {wall.capitalize()}: {coeff:.3f}")
    
    print(f"\nRIR Properties:")
    print(f"  Length: {len(rir)} samples")
    print(f"  Duration: {len(rir) / SAMPLING_RATE:.3f} seconds")
    print(f"  Peak amplitude: {torch.max(torch.abs(rir)):.6f}")
    print(f"  RMS level: {torch.sqrt(torch.mean(rir**2)):.6f}")


def example_2_small_dataset():
    """Example 2: Generate and analyze a small dataset."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Small Dataset Generation and Analysis")
    print("=" * 50)
    
    # Generate small dataset
    num_samples = 10
    print(f"Generating {num_samples} RIR samples...")
    
    dataset = []
    for i in range(num_samples):
        geometry_vector, rir = create_rir()
        dataset.append({'geom': geometry_vector, 'rir': rir})
        print(f"  Sample {i+1}/{num_samples} generated")
    
    # Analyze room volume distribution
    volumes = []
    for sample in dataset:
        geom = sample['geom'].numpy()
        room_dims = geom[:3]
        volume = np.prod(room_dims)
        volumes.append(volume)
    
    print(f"\nRoom Volume Statistics:")
    print(f"  Mean: {np.mean(volumes):.2f} m³")
    print(f"  Std: {np.std(volumes):.2f} m³")
    print(f"  Min: {np.min(volumes):.2f} m³")
    print(f"  Max: {np.max(volumes):.2f} m³")
    
    # Find most and least reverberant rooms (simple heuristic)
    rir_energies = [torch.sum(sample['rir']**2).item() for sample in dataset]
    most_reverb_idx = np.argmax(rir_energies)
    least_reverb_idx = np.argmin(rir_energies)
    
    print(f"\nReverberation Analysis (based on RIR energy):")
    print(f"  Most reverberant room: Sample {most_reverb_idx + 1}")
    print(f"  Least reverberant room: Sample {least_reverb_idx + 1}")


def example_3_custom_visualization():
    """Example 3: Custom visualization of RIR properties."""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Custom RIR Visualization")
    print("=" * 50)
    
    # Generate a few RIRs with different characteristics
    print("Generating RIRs with analysis...")
    
    rirs_data = []
    for i in range(3):
        geom, rir = create_rir()
        params = get_room_parameters_from_geometry(geom)
        
        # Calculate some basic metrics
        rir_np = rir.numpy()
        peak_idx = np.argmax(np.abs(rir_np))
        peak_time = peak_idx / SAMPLING_RATE
        
        # Simple RT60 estimation (very basic)
        rir_db = 20 * np.log10(np.abs(rir_np) + 1e-10)
        peak_db = np.max(rir_db)
        
        rirs_data.append({
            'rir': rir_np,
            'params': params,
            'peak_time': peak_time,
            'peak_db': peak_db
        })
    
    # Plot comparison
    fig, axes = plt.subplots(len(rirs_data), 1, figsize=(12, 4*len(rirs_data)))
    if len(rirs_data) == 1:
        axes = [axes]
    
    time_axis = np.arange(len(rirs_data[0]['rir'])) / SAMPLING_RATE
    
    for i, (ax, data) in enumerate(zip(axes, rirs_data)):
        ax.plot(time_axis, data['rir'])
        
        # Add room info to plot
        room_vol = np.prod(data['params']['room_dimensions'])
        avg_absorption = np.mean(list(data['params']['absorption_coefficients'].values()))
        
        title = f"RIR {i+1} - Volume: {room_vol:.1f}m³, Avg. Absorption: {avg_absorption:.3f}"
        ax.set_title(title)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show plot
    plot_path = get_plot_path('example_rir_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Custom visualization saved to: {plot_path}")
    plt.show()


def example_4_dataset_validation():
    """Example 4: Validate an existing dataset."""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Dataset Validation")
    print("=" * 50)
    
    dataset_path = get_output_path(DATASET_FILENAME)
    
    if Path(dataset_path).exists():
        print(f"Loading dataset from: {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"Dataset loaded successfully!")
        print(f"  Number of samples: {len(dataset)}")
        
        # Validate first few samples
        print("\nValidating sample structure...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            
            # Check keys
            expected_keys = {'geom', 'rir'}
            actual_keys = set(sample.keys())
            
            print(f"  Sample {i+1}:")
            print(f"    Keys present: {actual_keys == expected_keys}")
            print(f"    Geometry shape: {sample['geom'].shape}")
            print(f"    RIR shape: {sample['rir'].shape}")
            
            # Check for any NaN or infinite values
            geom_valid = torch.isfinite(sample['geom']).all()
            rir_valid = torch.isfinite(sample['rir']).all()
            
            print(f"    Geometry valid: {geom_valid}")
            print(f"    RIR valid: {rir_valid}")
    
    else:
        print(f"Dataset not found at: {dataset_path}")
        print("Run generate_rir_dataset_walls.py first to create a dataset.")


def example_5_batch_analysis():
    """Example 5: Batch analysis of absorption coefficient effects."""
    print("\n" + "=" * 50)
    print("EXAMPLE 5: Batch Analysis - Absorption Effects")
    print("=" * 50)
    
    print("Analyzing relationship between absorption and RIR characteristics...")
    
    # Generate samples with controlled absorption ranges
    low_absorption_samples = []
    high_absorption_samples = []
    
    num_samples_per_category = 5
    
    # Temporarily modify absorption range for controlled experiment
    original_absorption_range = ABSORPTION_RANGE
    
    for category, target_range in [("low", (0.1, 0.3)), ("high", (0.6, 0.9))]:
        print(f"\nGenerating {num_samples_per_category} samples with {category} absorption...")
        
        # Modify global absorption range temporarily
        import utils
        utils.ABSORPTION_RANGE = target_range
        
        samples = []
        for i in range(num_samples_per_category):
            geom, rir = create_rir()
            params = get_room_parameters_from_geometry(geom)
            
            # Calculate RIR energy as a simple reverberation metric
            rir_energy = torch.sum(rir**2).item()
            avg_absorption = np.mean(list(params['absorption_coefficients'].values()))
            
            samples.append({
                'avg_absorption': avg_absorption,
                'rir_energy': rir_energy,
                'params': params
            })
        
        if category == "low":
            low_absorption_samples = samples
        else:
            high_absorption_samples = samples
    
    # Restore original absorption range
    import utils
    utils.ABSORPTION_RANGE = original_absorption_range
    
    # Compare results
    low_energies = [s['rir_energy'] for s in low_absorption_samples]
    high_energies = [s['rir_energy'] for s in high_absorption_samples]
    
    print(f"\nResults:")
    print(f"Low absorption RIR energies - Mean: {np.mean(low_energies):.6f}, Std: {np.std(low_energies):.6f}")
    print(f"High absorption RIR energies - Mean: {np.mean(high_energies):.6f}, Std: {np.std(high_energies):.6f}")
    
    print(f"\nAs expected, lower absorption (more reflective walls) typically leads to")
    print(f"higher RIR energy due to longer reverberation times.")


def main():
    """Run all examples."""
    print("RIR Dataset Generator - Example Usage")
    print("This script demonstrates various features and capabilities.\n")
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_small_dataset()
        example_3_custom_visualization()
        example_4_dataset_validation()
        example_5_batch_analysis()
        # Small L-shaped dataset example (generate 10 samples quickly)
        print("\nEXAMPLE 6: Generate small L-shaped dataset (10 samples)")
        generate_l_shaped_dataset(num_rooms=10)
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the 'plots' directory for generated visualizations.")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        raise


if __name__ == "__main__":
    main()