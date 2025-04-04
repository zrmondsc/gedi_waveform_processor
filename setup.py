from setuptools import setup, find_packages

setup(
    name="gedi_waveform_processor",
    version="0.2",
    description="Process and export GEDI L1B waveforms for ML and DL",
    author="Zachary R Mondschein",
    author_email="mondschein.zr@gmail.com",
    url="https://github.com/zrmondsc/gedi_waveform_processor",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "geopandas",
        "h5py"
    ],
    extras_require={
        "tensorflow": ["tensorflow"],
        "torch": ["torch"]
    },
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
