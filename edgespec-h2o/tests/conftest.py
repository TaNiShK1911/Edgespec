# -*- coding: utf-8 -*-
"""Pytest configuration for EdgeSpec-H2O tests."""

import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-k not slow')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests if no CUDA available."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
