"""Setup script for Torch-Pruning package."""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Dependencies
requirements = ["torch>=2.0", "numpy", "graphviz"]

setuptools.setup(
    name="PAT",
    version="0.1.0",
    author="Avraham Raviv, Ishay Goldin",
    author_email="",
    description="Pruning-Aware Training on top of Torch-Pruning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AvrahamRaviv/sirc-torch-pruning",
    project_urls={
        "Bug Reports": "https://github.com/AvrahamRaviv/sirc-torch-pruning/issues",
        "Source": "https://github.com/AvrahamRaviv/sirc-torch-pruning",
    },
    packages=setuptools.find_packages(exclude=["tests*", "examples*", "reproduce*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=requirements,
    python_requires=">=3.7",
    keywords="pytorch, pruning, neural networks, deep learning, optimization",
    zip_safe=False,
)
