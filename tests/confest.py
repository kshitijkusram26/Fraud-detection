"""
tests/conftest.py
─────────────────
Pytest configuration — adds project root to sys.path
so all imports like `from src.xxx` work during testing.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))