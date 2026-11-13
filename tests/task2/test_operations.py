import hashlib
import io
import json
from pathlib import Path
import runpy
import tarfile
import pytest
from atml_pa3.task2 import runtime
from atml_pa3.task2.plotting import export_artifacts
from .test_evaluation import Backend,pairs
from atml_pa3.task2.evaluation import evaluate_final


def test_launcher_defaults_to_validation_without_model_or_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,"validate_assets",lambda root:{"identities":{}})
    monkeypatch.setattr(runtime,"make_spec",lambda *a:{"toy":True})
    def forbidden(*a,**k): raise AssertionError("Model must not load")
    monkeypatch.setattr(runtime,"load_vlm",forbidden)
    result=runtime.launch(tmp_path,tmp_path/"run")
    assert not result["training_started"] and not (tmp_path/"run").exists()
    monkeypatch.setattr(runtime.torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="requires CUDA"):
        runtime.launch(tmp_path,tmp_path/"run",execute=True)
    monkeypatch.setattr(runtime.torch.cuda,"is_available",lambda:True)
    from types import SimpleNamespace
    import atml_pa3.task2.batching as batching
    monkeypatch.setattr(batching,'detect_gpu',lambda *a:{'name':'6GB CUDA','total_bytes':6*1024**3,'free_bytes':5*1024**3})
    # A small GPU must reach actual component loading/probing, not a size gate.
    with pytest.raises(AssertionError,match='Model must not load'):
        runtime.launch(tmp_path,tmp_path/"run",execute=True)


def make_transfer(tmp_path,*,traversal=False):
    digest=lambda data:hashlib.sha256(data).hexdigest()
    content=b"fixture only"
    payload=json.dumps({"files":{"src/example.txt":{"bytes":len(content),"sha256":digest(content)}}}).encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode="w") as archive:
        for name,data in (("deployment_manifest.json",payload),("../escape" if traversal else "src/example.txt",content)):
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    blob=stream.getvalue();part=tmp_path/"part_0000.bin";part.write_bytes(blob)
    metadata={"parts":[{"name":part.name,"bytes":len(blob),"sha256":digest(blob)}],
              "archive_sha256":digest(blob),"deployment_manifest_sha256":digest(payload)}
    manifest=tmp_path/"transfer_manifest.json";manifest.write_text(json.dumps(metadata))
    return manifest


def unpacker():
    return runpy.run_path(str(Path(__file__).resolve().parents[2]/"scripts/task2_unpack.py"))["unpack"]


def test_chunk_reassembly_idempotent_and_tamper_detection(tmp_path):
    manifest=make_transfer(tmp_path);unpack=unpacker();destination=tmp_path/"task2-output"
    assert unpack(manifest,destination)["verified_files"]==1
    assert not unpack(manifest,destination)["training_started"]
    (destination/"src/example.txt").write_text("damaged")
    with pytest.raises(ValueError,match="payload differs"):unpack(manifest,destination)


def test_bad_chunk_and_archive_traversal_fail_closed(tmp_path):
    manifest=make_transfer(tmp_path)
    (tmp_path/"part_0000.bin").write_bytes(b"bad")
    with pytest.raises(ValueError,match="Chunk integrity"):unpacker()(manifest,tmp_path/"task2-output")
    nested=tmp_path/"nested";nested.mkdir();manifest=make_transfer(nested,traversal=True)
    with pytest.raises(ValueError,match="traversal"):unpacker()(manifest,nested/"task2-output")
    assert not (nested/"escape").exists()


def test_export_saved_results_only(tmp_path):
    final=tmp_path/"final"
    evaluate_final(Backend(),{"grounding":pairs()},final,model_identity="toy",dataset_identity="toy",resamples=20)
    result=export_artifacts(tmp_path/"plots",evaluation_dir=final)
    assert result==["grounding_conditions.png","grounding_pair_diagnostics.png","grounding_question_types.png"]
    assert (tmp_path/"plots/final_metrics.csv").exists()
    examples=json.loads((tmp_path/"plots/qualitative.json").read_text())
    assert len(examples["records"])==24
    assert not (tmp_path/"plots/training.csv").exists()
