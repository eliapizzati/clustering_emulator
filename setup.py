from setuptools import setup, find_packages

setup(
    name="clustering_emulator",
    version="0.1.0",
    description="Compute (cross-)correlation functions and their errors for cosmological simulations",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "Corrfunc",
    ],
)
