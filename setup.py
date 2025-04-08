# setup.py
from setuptools import setup, find_packages

setup(
    name="satellite_scheduling",
    version="0.1.0",
    description="A package for satellite network optimization",
    author="DMSH Team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "networkx",
    ],
    entry_points={
        'console_scripts': [
            'satellite-scheduler=satellite_scheduling.main:main',
        ],
    },
)

