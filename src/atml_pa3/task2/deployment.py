"""Local, hash-verified Task 2 packaging. No network or remote execution."""
from pathlib import Path
import shutil
import tarfile
from .io import atomic_json,sha256
from .runtime import validate_assets


def build_package(project,assets,output,*,chunk_bytes=128*1024**2):
    project,assets,output=Path(project).resolve(),Path(assets).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    validation=validate_assets(assets)
    files={}
    def include(path,relative):
        if not path.is_file():raise FileNotFoundError(path)
        files[relative.as_posix()]=path
    include(project/"src/atml_pa3/__init__.py",Path("src/atml_pa3/__init__.py"))
    for path in (project/"src/atml_pa3/task2").iterdir():
        if path.is_file():include(path,path.relative_to(project))
    for path in (project/"scripts").glob("task2_*"):
        if path.is_file():include(path,path.relative_to(project))
    for path in (project/"tests/task2").glob("*.py"):
        if path.name!="test_real_preparation.py":include(path,path.relative_to(project))
    for name in ("README.md","docs/task2/README.md","environment/task2-requirements.txt"):
        include(project/name,Path(name))
    for directory in ("data-v1","features-v1"):
        for path in (assets/directory).iterdir():
            if path.is_file():include(path,Path("artifacts/task2")/directory/path.name)
    for name in ("calibration.json","connector_initial_calibrated.pt"):
        include(assets/"vlm-validation-v1"/name,Path("artifacts/task2/vlm-validation-v1")/name)
    output.mkdir(parents=True)
    manifest={"version":1,"scope":"Task 2 only; no Task 1, raw images, archives, or trained model checkpoints",
              "asset_identities":validation["identities"],
              "files":{name:{"bytes":path.stat().st_size,"sha256":sha256(path)} for name,path in sorted(files.items())}}
    atomic_json(output/"deployment_manifest.json",manifest)
    archive=output/"task2.tar"
    temporary=output/"task2.tar.partial"
    with tarfile.open(temporary,"x") as bundle:
        bundle.add(output/"deployment_manifest.json",arcname="deployment_manifest.json")
        for name,path in sorted(files.items()):bundle.add(path,arcname=name,recursive=False)
    temporary.rename(archive)
    parts=[]
    with archive.open("rb") as source:
        index=0
        while True:
            block=source.read(chunk_bytes)
            if not block:break
            part=output/f"part_{index:04d}.bin"
            with part.open("xb") as out:out.write(block)
            parts.append({"name":part.name,"bytes":len(block),"sha256":sha256(part)})
            index+=1
    transfer={"archive":"task2.tar","archive_bytes":archive.stat().st_size,"archive_sha256":sha256(archive),
              "parts":parts,"deployment_manifest_sha256":sha256(output/"deployment_manifest.json")}
    atomic_json(output/"transfer_manifest.json",transfer)
    shutil.copyfile(project/"scripts/task2_unpack.py",output/"task2_unpack.py")
    return {"files":len(files),"archive_bytes":archive.stat().st_size,"parts":len(parts),"sha256":transfer["archive_sha256"]}
