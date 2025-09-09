import numpy as np
import torch
import pyroomacoustics as pra

# ---------- Parameters ----------
rir_length = 4096
fs = 16000
room_dim_range = [(3,10), (3,10), (2.5,4)]
absorption_range = [0.2, 0.8]

def create_rir():
    """
    Generate a single Room Impulse Response (RIR) and corresponding geometry vector.
    Each wall has a random absorption coefficient.
    """
    # Random room dimensions
    Lx = np.random.uniform(*room_dim_range[0])
    Ly = np.random.uniform(*room_dim_range[1])
    Lz = np.random.uniform(*room_dim_range[2])
    room_dim = [Lx, Ly, Lz]

    # Random wall absorption coefficients
    abs_coeffs = [np.random.uniform(*absorption_range) for _ in range(6)]
    wall_names = ['east', 'west', 'north', 'south', 'ceiling', 'floor']
    materials_dict = {name: pra.Material(a) for name, a in zip(wall_names, abs_coeffs)}

    # Create shoebox room
    room = pra.ShoeBox(room_dim, fs=fs, materials=materials_dict, max_order=15)

    # Random source and receiver positions
    src_pos = np.random.uniform([0,0,0], room_dim)
    rec_pos = np.random.uniform([0,0,0], room_dim)

    room.add_source(src_pos)
    room.add_microphone_array(pra.MicrophoneArray(rec_pos.reshape(3,1), room.fs))

    # Compute RIR
    room.compute_rir()
    rir = room.rir[0][0]
    rir = rir[:rir_length] if len(rir) >= rir_length else np.pad(rir, (0, rir_length-len(rir)))

    # Geometry vector
    geom_vector = np.concatenate([room_dim, src_pos, rec_pos, abs_coeffs])

    return torch.tensor(geom_vector, dtype=torch.float32), torch.tensor(rir, dtype=torch.float32)
