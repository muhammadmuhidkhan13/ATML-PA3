"""Fetch the pinned SigLIP snapshot into Task 2's own model cache."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from huggingface_hub import snapshot_download
from atml_pa3.task2.io import atomic_json, sha256

REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/task2/models"))
    args = parser.parse_args()
    path = Path(snapshot_download("google/siglip-base-patch16-224", revision=REVISION,
                                  cache_dir=args.root / "hub",
                                  allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"]))
    atomic_json(args.root / "siglip_snapshot.json",
                {"id": "google/siglip-base-patch16-224", "revision": REVISION,
                 "path": str(path.resolve()), "relative_path": path.resolve().relative_to(args.root.resolve()).as_posix(),
                 "files": {p.name: sha256(p) for p in path.iterdir() if p.is_file()}})
    print(path)
