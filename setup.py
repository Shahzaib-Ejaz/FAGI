from setuptools import setup, find_packages

setup(
    name="fagi",
    version="1.0.0",
    author="Shahzaib Ejaz",
    author_email="sejaz@student.na.edu",
    description="Federated Adversarial Graph Intelligence for Cloud API Threat Detection",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Shahzaib-Ejaz/FAGI",
    packages=find_packages(include=["fagi", "fagi.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "opacus>=1.4.0",
        "tqdm>=4.65.0",
    ],
    extras_require={"embeddings": ["transformers>=4.30.0"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
