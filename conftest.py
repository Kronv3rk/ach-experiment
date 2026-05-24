"""Pytest configuration: add the repository root to sys.path so tests can
import the `src` package without installation.
"""
import os
import sys

# Add the repository root (parent of this file) to sys.path
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
