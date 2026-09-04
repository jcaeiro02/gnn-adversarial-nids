from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gnn-adversarial-nids",
    version="0.1.0",
    author="João Caeiro",
    url="https://github.com/jcaeiro02/gnn-adversarial-nids",
    description="Adversarial robustness analysis of flow-centric GNN-based network intrusion detection systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "networkx",
        "pyyaml",
    ],
)