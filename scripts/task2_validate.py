import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atml_pa3.task2.validation import validate_prepared
from atml_pa3.task2.io import atomic_json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_prepared(args.data, args.sources)
    atomic_json(args.output, result)
    print(result)
