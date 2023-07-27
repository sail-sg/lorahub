# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lorahub",
    version="0.1",
    author="lorahub team",
    author_email="liuqian@sea.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sail-sg/lorahub",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'transformers',
        'peft',
        'nevergrad',
        'torch',
        'tqdm',
        'pandas',
        'numpy',
        'datasets'
    ],
)