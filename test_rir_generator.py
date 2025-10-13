#!/usr/bin/env python3
"""
Test suite for RIR Dataset Generator

This module contains unit tests for the RIR dataset generation functionality.
Run with: python -m pytest test_rir_generator.py
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

import numpy as np
import torch
import tempfile
import pickle
from pathlib import Path

# Import modules to test
from utils import create_rir, get_room_parameters_from_geometry
import config


class TestRIRGeneration:
    """Test cases for RIR generation functionality."""
    
    def test_create_rir_output_format(self):
        """Test that create_rir returns the expected output format."""
        geom_vector, rir = create_rir()
        
        # Check types
        assert isinstance(geom_vector, torch.Tensor), "Geometry vector should be a torch.Tensor"
        assert isinstance(rir, torch.Tensor), "RIR should be a torch.Tensor"
        
        # Check shapes
        assert geom_vector.shape == (15,), f"Geometry vector should have shape (15,), got {geom_vector.shape}"
        assert rir.shape == (config.RIR_LENGTH,), f"RIR should have shape ({config.RIR_LENGTH},), got {rir.shape}"
        
        # Check data types
        assert geom_vector.dtype == torch.float32, "Geometry vector should be float32"
        assert rir.dtype == torch.float32, "RIR should be float32"
    
    def test_create_rir_parameter_ranges(self):
        """Test that generated parameters are within expected ranges."""
        geom_vector, rir = create_rir()
        geom_np = geom_vector.numpy()
        
        # Check room dimensions
        room_dims = geom_np[:3]
        assert config.ROOM_DIM_RANGE[0][0] <= room_dims[0] <= config.ROOM_DIM_RANGE[0][1]
        assert config.ROOM_DIM_RANGE[1][0] <= room_dims[1] <= config.ROOM_DIM_RANGE[1][1]
        assert config.ROOM_DIM_RANGE[2][0] <= room_dims[2] <= config.ROOM_DIM_RANGE[2][1]
        
        # Check source positions (should be within room bounds)
        src_pos = geom_np[3:6]
        assert 0 <= src_pos[0] <= room_dims[0]
        assert 0 <= src_pos[1] <= room_dims[1]
        assert 0 <= src_pos[2] <= room_dims[2]
        
        # Check receiver positions (should be within room bounds)
        rec_pos = geom_np[6:9]
        assert 0 <= rec_pos[0] <= room_dims[0]
        assert 0 <= rec_pos[1] <= room_dims[1]
        assert 0 <= rec_pos[2] <= room_dims[2]
        
        # Check absorption coefficients
        abs_coeffs = geom_np[9:15]
        for coeff in abs_coeffs:
            assert config.ABSORPTION_RANGE[0] <= coeff <= config.ABSORPTION_RANGE[1]
    
    def test_create_rir_reproducibility(self):
        """Test that RIR generation produces different results each time (randomness)."""
        geom1, rir1 = create_rir()
        geom2, rir2 = create_rir()
        
        # Should be different (very unlikely to be identical due to randomness)
        assert not torch.allclose(geom1, geom2), "Generated geometry vectors should be different"
        assert not torch.allclose(rir1, rir2), "Generated RIRs should be different"
    
    def test_rir_properties(self):
        """Test basic properties of generated RIRs."""
        geom_vector, rir = create_rir()
        
        # RIR should not be all zeros
        assert torch.any(rir != 0), "RIR should not be all zeros"
        
        # RIR should have finite values
        assert torch.all(torch.isfinite(rir)), "RIR should have finite values"
        
        # Peak should be positive (convention for RIRs)
        peak_val = torch.max(torch.abs(rir))
        assert peak_val > 0, "RIR should have non-zero peak"
        
        # Peak should not be unreasonably large
        assert peak_val < 10.0, "RIR peak should be reasonable"


class TestParameterExtraction:
    """Test cases for parameter extraction functionality."""
    
    def test_get_room_parameters_from_geometry(self):
        """Test parameter extraction from geometry vector."""
        geom_vector, _ = create_rir()
        params = get_room_parameters_from_geometry(geom_vector)
        
        # Check structure
        expected_keys = {'room_dimensions', 'source_position', 'receiver_position', 'absorption_coefficients'}
        assert set(params.keys()) == expected_keys
        
        # Check dimensions
        assert len(params['room_dimensions']) == 3
        assert len(params['source_position']) == 3
        assert len(params['receiver_position']) == 3
        assert len(params['absorption_coefficients']) == 6
        
        # Check absorption coefficient keys
        expected_wall_names = set(config.WALL_NAMES)
        actual_wall_names = set(params['absorption_coefficients'].keys())
        assert actual_wall_names == expected_wall_names
    
    def test_parameter_consistency(self):
        """Test that extracted parameters match original geometry vector."""
        geom_vector, _ = create_rir()
        params = get_room_parameters_from_geometry(geom_vector)
        geom_np = geom_vector.numpy()
        
        # Check room dimensions
        np.testing.assert_array_almost_equal(params['room_dimensions'], geom_np[:3])
        
        # Check source position
        np.testing.assert_array_almost_equal(params['source_position'], geom_np[3:6])
        
        # Check receiver position
        np.testing.assert_array_almost_equal(params['receiver_position'], geom_np[6:9])
        
        # Check absorption coefficients
        abs_coeffs_list = [params['absorption_coefficients'][name] for name in config.WALL_NAMES]
        np.testing.assert_array_almost_equal(abs_coeffs_list, geom_np[9:15])


class TestConfiguration:
    """Test cases for configuration validation."""
    
    def test_config_validation(self):
        """Test that configuration validation works."""
        # This should pass with default configuration
        assert config.validate_config() == True
    
    def test_config_constants(self):
        """Test that configuration constants are reasonable."""
        assert config.RIR_LENGTH > 0
        assert config.SAMPLING_RATE > 0
        assert config.NUM_ROOMS > 0
        
        # Check absorption range
        assert 0 <= config.ABSORPTION_RANGE[0] <= 1
        assert 0 <= config.ABSORPTION_RANGE[1] <= 1
        assert config.ABSORPTION_RANGE[0] < config.ABSORPTION_RANGE[1]
        
        # Check room dimension ranges
        for dim_range in config.ROOM_DIM_RANGE:
            assert len(dim_range) == 2
            assert dim_range[0] > 0
            assert dim_range[0] < dim_range[1]


class TestDatasetOperations:
    """Test cases for dataset-level operations."""
    
    def test_small_dataset_generation(self):
        """Test generation of a small dataset."""
        num_samples = 5
        dataset = []
        
        for i in range(num_samples):
            geom, rir = create_rir()
            dataset.append({'geom': geom, 'rir': rir})
        
        assert len(dataset) == num_samples
        
        # Check each sample
        for sample in dataset:
            assert 'geom' in sample
            assert 'rir' in sample
            assert isinstance(sample['geom'], torch.Tensor)
            assert isinstance(sample['rir'], torch.Tensor)
    
    def test_dataset_serialization(self):
        """Test that datasets can be saved and loaded."""
        # Generate small dataset
        dataset = []
        for i in range(3):
            geom, rir = create_rir()
            dataset.append({'geom': geom, 'rir': rir})
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(dataset, f)
            temp_path = f.name
        
        try:
            # Load from file
            with open(temp_path, 'rb') as f:
                loaded_dataset = pickle.load(f)
            
            # Check that loaded dataset matches original
            assert len(loaded_dataset) == len(dataset)
            
            for orig, loaded in zip(dataset, loaded_dataset):
                assert torch.allclose(orig['geom'], loaded['geom'])
                assert torch.allclose(orig['rir'], loaded['rir'])
        
        finally:
            # Clean up
            Path(temp_path).unlink()


def run_basic_tests():
    """Run a subset of tests without pytest."""
    print("Running basic tests...")
    
    try:
        # Test RIR generation
        print("  Testing RIR generation...")
        geom, rir = create_rir()
        assert geom.shape == (15,)
        assert rir.shape == (config.RIR_LENGTH,)
        print("    ✓ RIR generation works")
        
        # Test parameter extraction
        print("  Testing parameter extraction...")
        params = get_room_parameters_from_geometry(geom)
        assert len(params['room_dimensions']) == 3
        print("    ✓ Parameter extraction works")
        
        # Test configuration
        print("  Testing configuration...")
        assert config.validate_config()
        print("    ✓ Configuration validation works")
        
        print("All basic tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run basic tests if pytest is not available
    success = run_basic_tests()
    
    if not success:
        exit(1)
    
    print("\nFor comprehensive testing, install pytest and run:")
    print("  pip install pytest")
    print("  python -m pytest test_rir_generator.py -v")