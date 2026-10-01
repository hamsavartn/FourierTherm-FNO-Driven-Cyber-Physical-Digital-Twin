"""Make the src/ package importable for pytest runs from the repo root."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
