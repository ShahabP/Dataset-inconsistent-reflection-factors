#!/usr/bin/env python3
"""
Setup script for RIR Dataset Generator

This setup script allows for easy installation of the RIR Dataset Generator package
and its dependencies.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                # Extract package name without version constraints for setup.py
                package = line.split('>=')[0].split('==')[0].split('<')[0]
                if package:
                    requirements.append(package)

setup(
    name="rir-dataset-generator",
    version="1.0.0",
    author="Shahab Pasha",
    author_email="your.email@example.com",
    description="Generate Room Impulse Response datasets with inconsistent wall reflection factors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShahabP/Dataset-inconsistent-reflection-factors",
    project_urls={
        "Bug Reports": "https://github.com/ShahabP/Dataset-inconsistent-reflection-factors/issues",
        "Source": "https://github.com/ShahabP/Dataset-inconsistent-reflection-factors",
        "Documentation": "https://github.com/ShahabP/Dataset-inconsistent-reflection-factors/blob/main/README.md",
    },
    
    # Package information
    py_modules=[
        "utils",
        "config", 
        "generate_rir_dataset_walls",
        "plot_rirs",
        "examples"
    ],
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.900',
        ],
        'analysis': [
            'scipy>=1.7.0',
            'scikit-learn>=1.0.0',
            'librosa>=0.9.0',
            'soundfile>=0.10.0',
        ],
        'data': [
            'pandas>=1.3.0',
            'h5py>=3.6.0',
        ],
    },
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'generate-rir-dataset=generate_rir_dataset_walls:main',
            'plot-rirs=plot_rirs:main',
            'rir-examples=examples:main',
        ],
    },
    
    # Classifiers help users find your project
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Keywords for PyPI search
    keywords=[
        "room-impulse-response", 
        "acoustic-simulation", 
        "audio-processing",
        "machine-learning",
        "dataset-generation",
        "reverberation",
        "speech-enhancement",
        "pyroomacoustics"
    ],
    
    # Include additional files
    include_package_data=True,
    package_data={
        '': ['README.md', 'requirements.txt', 'LICENSE'],
    },
    
    # Zip safety
    zip_safe=False,
)