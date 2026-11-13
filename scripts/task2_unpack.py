"""Standalone safe chunk reassembly/extraction; never starts training."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import uuid

def digest(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):result.update(block)
    return result.hexdigest()

def unpack(manifest_path,output):
    manifest_path,output=Path(manifest_path).resolve(),Path(output).resolve()
    if "task2" not in output.name.lower() or output==output.parent:
        raise ValueError("Extract into a distinct explicitly named Task2 directory")
    metadata=json.loads(manifest_path.read_text())
    archive=manifest_path.parent/"task2.tar"
    if not archive.exists():
        temporary=archive.with_suffix(".partial")
        with temporary.open("wb") as out:
            for part in metadata["parts"]:
                if Path(part["name"]).name!=part["name"]:raise ValueError("Invalid chunk path")
                path=manifest_path.parent/part["name"]
                if path.stat().st_size!=part["bytes"] or digest(path)!=part["sha256"]:raise ValueError("Chunk integrity failure")
                with path.open("rb") as source:
                    for block in iter(lambda:source.read(1024*1024),b""):out.write(block)
        if digest(temporary)!=metadata["archive_sha256"]:raise ValueError("Reassembled archive hash differs")
        temporary.replace(archive)
    if digest(archive)!=metadata["archive_sha256"]:raise ValueError("Archive integrity failure")
    target=output if output.exists() else output.parent/(".unpacking-task2-"+uuid.uuid4().hex)
    if target!=output:
        target.mkdir(parents=True)
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                destination=(target/member.name).resolve()
                if destination!=target and target not in destination.parents:raise ValueError("Archive traversal")
                if not member.isfile():raise ValueError("Only regular payload files are allowed")
                destination.parent.mkdir(parents=True,exist_ok=True)
                with bundle.extractfile(member) as source,destination.open("xb") as out:
                    for block in iter(lambda:source.read(1024*1024),b""):out.write(block)
    manifest=target/"deployment_manifest.json"
    if digest(manifest)!=metadata["deployment_manifest_sha256"]:raise ValueError("Deployment manifest hash differs")
    data=json.loads(manifest.read_text())
    for name,expected in data["files"].items():
        path=(target/name).resolve()
        if target not in path.parents:raise ValueError("Manifest traversal")
        if path.stat().st_size!=expected["bytes"] or digest(path)!=expected["sha256"]:raise ValueError(f"Extracted payload differs: {name}")
    if target!=output:
        if output.exists():raise FileExistsError(output)
        target.rename(output)
    return {"verified_files":len(data["files"]),"output":str(output),"training_started":False}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();print(unpack(a.manifest,a.output))
