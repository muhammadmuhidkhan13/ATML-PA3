"""Targeted COCO image acquisition with verification and restart support."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from .io import atomic_json, download, read_json, sha256


def acquire_images(data_dir, root, workers=8):
    data_dir, root = Path(data_dir), Path(root)
    images = read_json(data_dir / "images.json")
    def fetch(item):
        key, image = item
        path = root / image["relative_path"]
        url = f"https://s3.amazonaws.com/images.cocodataset.org/{image['source_split']}/{image['file_name']}"
        record = download(url, path)
        with Image.open(path) as loaded:
            if loaded.size != (image["width"], image["height"]):
                raise ValueError(f"Image dimensions disagree with COCO: {key}")
            loaded.verify()
        return key, {"relative_path": image["relative_path"], "sha256": record["sha256"],
                     "source_split": image["source_split"], "url": url}
    records, errors = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, item): item[0] for item in images.items()}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                key, record = future.result()
                records[key] = record
            except Exception as exc:
                errors[futures[future]] = str(exc)
            if i % 100 == 0:
                print(f"Images verified {i}/{len(images)}; failures {len(errors)}", flush=True)
    manifest = {"selected_images_sha256": sha256(data_dir / "images.json"), "images": records,
                "errors": errors, "complete": not errors and len(records) == len(images)}
    atomic_json(root / "image_manifest.json", manifest)
    if errors:
        raise RuntimeError(f"{len(errors)} required images unavailable; see image_manifest.json")
    return {"images": len(records), "complete": True}
