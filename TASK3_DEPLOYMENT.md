# Task 3 live deployment

The original v6 live run completed on an RTX 3090. Use this guide for a fresh deployment of the packaged code. In a full repository checkout, see [Task 3](docs/task3/README.md) for the completed evidence and CPU verification commands.

Use one dedicated supported GPU throughout. V6 accepts an RTX 3090 or RTX 3090 Ti with at least 23 GiB reported capacity, or an A100 with at least 70 GiB reported capacity. All require at least 12 GiB free and at most 4 GiB already occupied before model load. The reference stack is Linux, Python 3.12.15, PyTorch 2.14.1+cu130 and the auxiliary pins in the bundle's requirements file.

From the extracted bundle root, verify the manifest and install the auxiliary requirements into a dedicated environment with the intended CUDA PyTorch already available.

```sh
python scripts/task3_bundle_verify.py --bundle .
python -m pip install -r requirements.txt
python -m pytest tests/task3 -q
hf download Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --local-dir model
python scripts/task3_gpu_preflight.py --assets assets --model-path model --output runs/preflight-v6 --execute-preflight
python scripts/task3_live.py --assets assets --model-path model --preflight runs/preflight-v6 --output runs/live-v1 --execute-generation
python scripts/task3_live_export.py --assets assets --run runs/live-v1 --output exports/live-v1
```

Preflight makes three synthetic calls, with memory probes that reach 4,096 and 8,192 input tokens and can exceed those targets by up to 256 tokens. It requires at least 2 GiB free after each probe. Official generation enforces the 4,096- and 8,192-token rendered input caps. Use a fresh output directory for a new preflight or export. To resume generation, repeat the same command against the existing run and proof on the same GPU and software stack. See [live record protocol](TASK3_LIVE_INFERENCE.md).
