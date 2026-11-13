import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atml_pa3.task2.features import build_cache
from atml_pa3.task2.io import read_json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    metadata = read_json(args.root / "models/siglip_snapshot.json")
    path = args.model_path or args.root / "models" / metadata["relative_path"]
    print(build_cache(args.root, args.cache, path, args.batch_size))
