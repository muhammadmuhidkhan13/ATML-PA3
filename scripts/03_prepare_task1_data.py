"""Portable entry point for the Task 1 data dry run."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atml_pa3.task1.prepare import main

if __name__ == "__main__":
    main()
