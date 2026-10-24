#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup for EdgeSpec-H2O."""

from setuptools import setup, find_packages

setup(
    name='edgespec-h2o',
    version='0.1.0',
    description='EdgeSpec-H2O: Low-RAM INT4 inference for Jetson Nano',
    packages=find_packages(exclude=['tests*', 'tools*', 'bench*']),
    python_requires='>=3.6',
    install_requires=[
        'torch',
    ],
)
