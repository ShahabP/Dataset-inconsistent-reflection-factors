#!/usr/bin/env python3
"""
RIR Dataset Generator with Inconsistent Wall Reflection Factors

This script generates a large-scale dataset of Room Impulse Responses (RIRs) where each room
has different and inconsistent wall absorption coefficients. The dataset is designed for
research applications in speech enhancement, dereverberation, and acoustic modeling.

Usage:
    python generate_rir_dataset_walls.py

Output:
    - rir_dataset_walls.pkl: Pickle file containing the generated dataset
    
The dataset contains 5000 room configurations, each with:
    - Unique room dimensions, source/receiver positions
    - Independent absorption coefficients for all 6 wall surfaces
    - 4096-sample RIR at 16 kHz sampling rate
"""

import pickle
import time
from typing import List, Dict, Any
import sys
import os

from utils import create_rir

# ---------- Configuration Parameters ----------
NUM_ROOMS: int = 5000          # Total number of room configurations to generate
OUTPUT_FILENAME: str = 'rir_dataset_walls.pkl'  # Output dataset filename
PROGRESS_INTERVAL: int = 100   # Print progress every N rooms


def generate_rir_dataset(num_rooms: int = NUM_ROOMS, 
                        output_file: str = OUTPUT_FILENAME,
                        progress_interval: int = PROGRESS_INTERVAL) -> None:
    """
    Generate a dataset of Room Impulse Responses with inconsistent wall properties.
    
    Args:
        num_rooms (int): Number of room configurations to generate
        output_file (str): Filename for the output pickle file
        progress_interval (int): Interval for printing progress updates
        
    Returns:
        None
    """
    print("=" * 60)
    print("RIR Dataset Generator - Inconsistent Wall Reflection Factors")
    print("=" * 60)
    print(f"Generating {num_rooms:,} room configurations...")
    print(f"Output file: {output_file}")
    print(f"Progress updates every {progress_interval} rooms")
    print()
    
    dataset: List[Dict[str, Any]] = []
    start_time = time.time()
    
    try:
        for i in range(num_rooms):
            # Generate RIR and geometry for current room
            geometry_vector, rir = create_rir()
            
            # Store in dataset with descriptive keys
            room_data = {
                'geom': geometry_vector,  # 15-element geometry vector
                'rir': rir               # 4096-sample impulse response
            }
            dataset.append(room_data)
            
            # Print progress updates
            if (i + 1) % progress_interval == 0:
                elapsed_time = time.time() - start_time
                rooms_per_second = (i + 1) / elapsed_time
                eta_seconds = (num_rooms - (i + 1)) / rooms_per_second if rooms_per_second > 0 else 0
                
                print(f"Progress: {i + 1:,}/{num_rooms:,} rooms generated "
                      f"({(i + 1) / num_rooms * 100:.1f}%) - "
                      f"Rate: {rooms_per_second:.1f} rooms/sec - "
                      f"ETA: {eta_seconds / 60:.1f} min")
        
        # Save the complete dataset
        print("\nSaving dataset...")
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Print completion summary
        total_time = time.time() - start_time
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print("\n" + "=" * 60)
        print("Dataset Generation Complete!")
        print("=" * 60)
        print(f"Total rooms generated: {num_rooms:,}")
        print(f"Total generation time: {total_time / 60:.1f} minutes")
        print(f"Average rate: {num_rooms / total_time:.1f} rooms/second")
        print(f"Output file: {output_file}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Memory usage per room: ~{file_size_mb * 1024 / num_rooms:.1f} KB")
        
    except KeyboardInterrupt:
        print(f"\n\nGeneration interrupted by user after {len(dataset)} rooms.")
        print("Saving partial dataset...")
        
        # Save partial dataset with different filename
        partial_filename = f"partial_{output_file}"
        with open(partial_filename, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Partial dataset saved as: {partial_filename}")
        print(f"Rooms generated: {len(dataset)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError occurred during generation: {str(e)}")
        print("This might be due to memory constraints or invalid parameters.")
        sys.exit(1)


def validate_dataset(dataset_file: str = OUTPUT_FILENAME) -> None:
    """
    Validate the generated dataset by checking its structure and content.
    
    Args:
        dataset_file (str): Path to the dataset pickle file
    """
    try:
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"\nDataset Validation Results:")
        print(f"- Total samples: {len(dataset):,}")
        print(f"- First sample keys: {list(dataset[0].keys())}")
        print(f"- Geometry vector shape: {dataset[0]['geom'].shape}")
        print(f"- RIR shape: {dataset[0]['rir'].shape}")
        print(f"- Geometry vector type: {type(dataset[0]['geom'])}")
        print(f"- RIR type: {type(dataset[0]['rir'])}")
        
        # Check for data consistency
        geom_shapes = [sample['geom'].shape for sample in dataset[:100]]  # Check first 100
        rir_shapes = [sample['rir'].shape for sample in dataset[:100]]
        
        print(f"- Geometry shapes consistent: {len(set(geom_shapes)) == 1}")
        print(f"- RIR shapes consistent: {len(set(rir_shapes)) == 1}")
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")


def main():
    """Main function for command-line execution."""
    generate_rir_dataset()
    validate_dataset()

if __name__ == "__main__":
    main()
