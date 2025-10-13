"""
Utility functions for generating Room Impulse Responses (RIRs) with inconsistent wall reflection factors.

This module provides the core functionality for creating synthetic RIRs in shoebox-shaped rooms
where each wall has independent, randomized absorption coefficients to simulate realistic
acoustic environments.
"""

from typing import Tuple, List
import numpy as np
import torch
import pyroomacoustics as pra

# ---------- Configuration Parameters ----------
# These parameters control the characteristics of generated RIRs
RIR_LENGTH: int = 4096  # Length of generated RIR in samples
SAMPLING_RATE: int = 16000  # Sampling rate in Hz

# Room dimension ranges (min, max) in meters: [x, y, z]
ROOM_DIM_RANGE: List[Tuple[float, float]] = [(3.0, 10.0), (3.0, 10.0), (2.5, 4.0)]

# Absorption coefficient range (min, max) for wall materials
ABSORPTION_RANGE: Tuple[float, float] = (0.2, 0.8)

# Maximum order of reflections for image source method
MAX_REFLECTION_ORDER: int = 15

# Wall names for material assignment
WALL_NAMES: List[str] = ['east', 'west', 'north', 'south', 'ceiling', 'floor']


def create_rir() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a single Room Impulse Response (RIR) and corresponding geometry vector.
    
    This function creates a synthetic RIR by simulating a shoebox-shaped room where each
    wall has an independent, randomly assigned absorption coefficient. The function also
    generates random source and receiver positions within the room.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - geom_vector (torch.Tensor): Geometry vector of shape (15,) containing:
                [room_x, room_y, room_z, src_x, src_y, src_z, rec_x, rec_y, rec_z,
                 abs_east, abs_west, abs_north, abs_south, abs_ceiling, abs_floor]
            - rir (torch.Tensor): Room impulse response of shape (RIR_LENGTH,)
    
    Note:
        The geometry vector contains all the parameters needed to reproduce the RIR:
        - Elements 0-2: Room dimensions [Lx, Ly, Lz] in meters
        - Elements 3-5: Source position [x, y, z] in meters  
        - Elements 6-8: Receiver position [x, y, z] in meters
        - Elements 9-14: Wall absorption coefficients [east, west, north, south, ceiling, floor]
    """
    # Generate random room dimensions within specified ranges
    room_x = np.random.uniform(*ROOM_DIM_RANGE[0])
    room_y = np.random.uniform(*ROOM_DIM_RANGE[1])
    room_z = np.random.uniform(*ROOM_DIM_RANGE[2])
    room_dimensions = [room_x, room_y, room_z]

    # Generate independent absorption coefficients for each wall
    absorption_coefficients = [
        np.random.uniform(*ABSORPTION_RANGE) for _ in range(len(WALL_NAMES))
    ]
    
    # Create material dictionary mapping wall names to pyroomacoustics Material objects
    materials_dict = {
        wall_name: pra.Material(absorption_coeff)
        for wall_name, absorption_coeff in zip(WALL_NAMES, absorption_coefficients)
    }

    # Create shoebox room with specified materials and acoustic properties
    room = pra.ShoeBox(
        room_dimensions,
        fs=SAMPLING_RATE,
        materials=materials_dict,
        max_order=MAX_REFLECTION_ORDER
    )

    # Generate random source and receiver positions within the room bounds
    source_position = np.random.uniform([0, 0, 0], room_dimensions)
    receiver_position = np.random.uniform([0, 0, 0], room_dimensions)

    # Add source and microphone to the room
    room.add_source(source_position)
    room.add_microphone_array(
        pra.MicrophoneArray(receiver_position.reshape(3, 1), room.fs)
    )

    # Compute the room impulse response using image source method
    room.compute_rir()
    rir_raw = room.rir[0][0]  # Extract RIR for first microphone from first source
    
    # Ensure RIR has the correct length by truncating or zero-padding
    if len(rir_raw) >= RIR_LENGTH:
        rir_processed = rir_raw[:RIR_LENGTH]
    else:
        rir_processed = np.pad(rir_raw, (0, RIR_LENGTH - len(rir_raw)), mode='constant')

    # Create comprehensive geometry vector containing all room parameters
    geometry_vector = np.concatenate([
        room_dimensions,           # Room dimensions (3 elements)
        source_position,           # Source position (3 elements)  
        receiver_position,         # Receiver position (3 elements)
        absorption_coefficients    # Wall absorption coefficients (6 elements)
    ])

    # Convert to PyTorch tensors with appropriate data types
    return (
        torch.tensor(geometry_vector, dtype=torch.float32),
        torch.tensor(rir_processed, dtype=torch.float32)
    )


def get_room_parameters_from_geometry(geom_vector: torch.Tensor) -> dict:
    """
    Extract room parameters from a geometry vector.
    
    Args:
        geom_vector (torch.Tensor): Geometry vector of shape (15,)
        
    Returns:
        dict: Dictionary containing room parameters with keys:
            'room_dimensions', 'source_position', 'receiver_position', 'absorption_coefficients'
    """
    geom_np = geom_vector.numpy() if isinstance(geom_vector, torch.Tensor) else geom_vector
    
    return {
        'room_dimensions': geom_np[0:3].tolist(),
        'source_position': geom_np[3:6].tolist(),
        'receiver_position': geom_np[6:9].tolist(),
        'absorption_coefficients': {
            wall_name: float(geom_np[9 + i])
            for i, wall_name in enumerate(WALL_NAMES)
        }
    }
