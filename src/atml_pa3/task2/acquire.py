"""Acquire official metadata; images are selected and fetched separately."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import re
import zipfile

from .io import atomic_json, download, sha256, read_json

VQA_BASE = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
SOURCES = {"coco": "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip"}
for split, label in (("train", "Train"), ("val", "Val")):
    for kind in ("Annotations", "Questions", "Complementary_Pairs"):
        SOURCES[f"vqa_{split}_{kind.lower()}"] = VQA_BASE + f"v2_{kind}_{label}_mscoco.zip"


def acquire(root: Path):
    import requests
    root = Path(root)
    records = {}
    for name, url in SOURCES.items():
        print(f"Acquiring {name}...", flush=True)
        archive = root / "downloads" / url.rsplit("/", 1)[1]
        records[name] = download(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                basename = Path(member).name
                if not basename.endswith(".json"):
                    continue
                target = root / "annotations" / basename
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    temp = target.with_suffix(".partial")
                    with bundle.open(member) as source, temp.open("wb") as out:
                        import shutil
                        shutil.copyfileobj(source, out)
                    temp.replace(target)
                records[name].setdefault("extracted", {})[basename] = sha256(target)
        atomic_json(root / "source_manifest.json", records)
    # Pin and retain the official normalization source; only literal lookup
    # tables are extracted, never execute downloaded Python.
    source = root / "normalization" / "vqaEval.py"
    sidecar = source.with_suffix(".py.source.json")
    if sidecar.exists():
        revision = read_json(sidecar)["url"].split("/VQA/")[1].split("/")[0]
    else:
        response = requests.get("https://api.github.com/repos/GT-Vision-Lab/VQA/commits/master", timeout=30)
        response.raise_for_status()
        revision = response.json()["sha"]
    url = f"https://raw.githubusercontent.com/GT-Vision-Lab/VQA/{revision}/PythonEvaluationTools/vqaEvaluation/vqaEval.py"
    record = download(url, source)
    text = source.read_text(encoding="utf-8")
    tables = {}
    for name in ("contractions", "manualMap", "articles", "punct"):
        match = re.search(r"self\." + name + r"\s*=\s*", text)
        if not match:
            raise ValueError(f"Official normalization table missing: {name}")
        lines = text[match.end():].splitlines(keepends=True)
        for end in range(1, len(lines) + 1):
            try:
                tables[name] = ast.literal_eval("".join(lines[:end]).strip())
                break
            except (SyntaxError, ValueError):
                continue
        else:
            raise ValueError(f"Invalid normalization literal: {name}")
    tables["source"] = {"revision": revision, **record}
    atomic_json(Path(__file__).parent / "vqa_tables.json", tables)
    records["normalization"] = tables["source"]
    atomic_json(root / "source_manifest.json", records)
    print("Official metadata acquisition complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    acquire(parser.parse_args().root)
