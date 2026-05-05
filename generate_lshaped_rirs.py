"""
Generate RIRs for simple L-shaped (non-convex) rooms.

This script produces a small, simple dataset of synthetic Room Impulse
Responses (RIRs) for concave L-shaped rooms using pyroomacoustics' polygon
room support (2D) and extruding to 3D when available. It is intentionally
simple and meant for dataset augmentation and experimentation.

The script saves a pickle containing tuples of (geometry_vector, rir_tensor).
"""
from typing import List, Tuple
import os
import pickle
import numpy as np
import torch
import pyroomacoustics as pra
from matplotlib.path import Path

import config


def make_l_shape_vertices(Lx: float, Ly: float, cut_w: float, cut_h: float) -> List[Tuple[float, float]]:
    """Return CCW vertices for a simple L-shaped polygon in the XY plane.

    The shape is a rectangle of size (Lx, Ly) with a rectangular cutout
    of size (cut_w, cut_h) removed from the top-right corner.
    """
    # Ensure cut sizes are valid
    cut_w = min(max(0.1, cut_w), Lx - 0.2)
    cut_h = min(max(0.1, cut_h), Ly - 0.2)

    verts = [
        (0.0, 0.0),
        (Lx, 0.0),
        (Lx, Ly - cut_h),
        (Lx - cut_w, Ly - cut_h),
        (Lx - cut_w, Ly),
        (0.0, Ly),
    ]
    return verts


def sample_point_in_polygon(verts: List[Tuple[float, float]]) -> Tuple[float, float]:
    path = Path(verts)
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # rejection sampling inside bounding box
    for _ in range(1000):
        x = np.random.uniform(minx + 0.05, maxx - 0.05)
        y = np.random.uniform(miny + 0.05, maxy - 0.05)
        if path.contains_point((x, y)):
            return x, y
    # fallback to centroid
    centroid = np.mean(np.array(verts), axis=0)
    return float(centroid[0]), float(centroid[1])


def create_rir_for_l_shape() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate one RIR and geometry vector for an L-shaped room.

    Geometry vector layout (simple):
      [Lx, Ly, cut_w, cut_h, height, src_x, src_y, src_z, rec_x, rec_y, rec_z, absorption]
    """
    # Room size parameters sampled from config ranges
    Lx = np.random.uniform(*config.ROOM_DIM_RANGE[0])
    Ly = np.random.uniform(*config.ROOM_DIM_RANGE[1])
    height = np.random.uniform(*config.ROOM_DIM_RANGE[2])

    # Cutout sizes to create L-shape (fraction of room)
    cut_w = np.random.uniform(0.2 * Lx, 0.5 * Lx)
    cut_h = np.random.uniform(0.2 * Ly, 0.5 * Ly)

    verts = make_l_shape_vertices(Lx, Ly, cut_w, cut_h)

    # Absorption for the whole room (very simple single-value model)
    absorption = float(np.random.uniform(*config.ABSORPTION_RANGE))
    material = pra.Material(absorption)

    # Build a 2D polygon room and extrude to 3D if supported
    try:
        corners = np.array(verts).T
        room = pra.Room.from_corners(corners, fs=config.SAMPLING_RATE, materials=material, max_order=config.MAX_REFLECTION_ORDER)
        # some pyroomacoustics versions allow extrude
        if hasattr(room, "extrude"):
            room.extrude(height)
    except Exception:
        # Fallback: use a bounding shoebox but treat concavity implicitly
        room = pra.ShoeBox([Lx, Ly, height], fs=config.SAMPLING_RATE, materials=material, max_order=config.MAX_REFLECTION_ORDER)

    # Sample source and receiver positions inside L polygon (z inside height)
    sx, sy = sample_point_in_polygon(verts)
    rx, ry = sample_point_in_polygon(verts)
    sz = np.random.uniform(0.2, height - 0.2)
    rz = np.random.uniform(0.2, height - 0.2)

    src = np.array([sx, sy, sz])
    rec = np.array([rx, ry, rz])

    room.add_source(src)
    room.add_microphone_array(pra.MicrophoneArray(rec.reshape(3, 1), room.fs))

    room.compute_rir()
    rir_raw = room.rir[0][0]

    # truncate or pad
    if len(rir_raw) >= config.RIR_LENGTH:
        rir = np.asarray(rir_raw[: config.RIR_LENGTH])
    else:
        rir = np.pad(rir_raw, (0, config.RIR_LENGTH - len(rir_raw)), mode="constant")

    geom = np.array([Lx, Ly, cut_w, cut_h, height, sx, sy, sz, rx, ry, rz, absorption], dtype=np.float32)

    return torch.tensor(geom, dtype=torch.float32), torch.tensor(rir, dtype=torch.float32)


def generate_dataset(num_rooms: int = None, out_path: str = None):
    if num_rooms is None:
        num_rooms = config.NUM_ROOMS
    if out_path is None:
        out_path = config.get_output_path("rir_dataset_l_shaped.pkl")

    dataset = []
    for i in range(num_rooms):
        geom, rir = create_rir_for_l_shape()
        dataset.append((geom, rir))
        if (i + 1) % config.PROGRESS_INTERVAL == 0:
            print(f"Generated {i+1}/{num_rooms} L-shaped RIRs")

    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=config.PICKLE_PROTOCOL)

    print(f"Saved {len(dataset)} L-shaped RIRs to {out_path}")


if __name__ == "__main__":
    generate_dataset()
