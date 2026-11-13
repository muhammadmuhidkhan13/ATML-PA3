"""Analyze the official LoRA-16 effective updates on CPU."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


MEMBER = "artifacts/task1/official-lora16-v1/checkpoints/update_01250/state.pt"
SHA256 = "ec24c4d92006a8bc70371d028b0cdb0f1e3dbd50548aba812a22891e80241e5a"
RANK = 16
ALPHA = 32
LAYERS = 28
# Dimensions are (output, input) for the pinned Qwen3-0.6B architecture.
TARGETS = {
    "q_proj": ("self_attn", 2048, 1024),
    "k_proj": ("self_attn", 1024, 1024),
    "v_proj": ("self_attn", 1024, 1024),
    "o_proj": ("self_attn", 1024, 2048),
    "gate_proj": ("mlp", 3072, 1024),
    "up_proj": ("mlp", 3072, 1024),
    "down_proj": ("mlp", 1024, 3072),
}


def load_state(path):
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r|*") as archive:
            for member in archive:
                if member.name.removeprefix("./") == MEMBER and member.isfile():
                    data = archive.extractfile(member).read()
                    break
            else:
                raise ValueError(f"Missing official checkpoint member {MEMBER}")
    else:
        data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Checkpoint SHA-256 differs from the official LoRA-16 final record")
    # The pinned training payload includes NumPy RNG state. Deserialize only
    # after verifying the exact digest of this trusted local checkpoint.
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    if payload["metadata"]["completed_updates"] != 1250:
        raise ValueError("Expected the checkpoint at update 1250")
    return payload["model"]


def spectrum(a, b, scale):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]
            or not 0 < a.shape[0] <= min(a.shape[1], b.shape[0])):
        raise ValueError("Incompatible LoRA factors")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Non-finite LoRA factor")
    _, rb = np.linalg.qr(b, mode="reduced")
    _, ra = np.linalg.qr(a.T, mode="reduced")
    sigma = np.linalg.svd(scale * (rb @ ra.T), compute_uv=False)
    energy = sigma**2
    total = float(energy.sum())
    # Independent norm identity, without forming the dense update.
    gram_products = (a @ a.T) * (b.T @ b)
    gram_norm_sq = scale**2 * gram_products.sum()
    gram_scale = scale**2 * np.abs(gram_products).sum()
    if not np.isclose(total, gram_norm_sq, rtol=1e-10, atol=1e-14 * gram_scale):
        raise ValueError("Spectral energy disagrees with the factor Gram norm")
    tolerance = max(a.shape[1], b.shape[0]) * np.finfo(np.float64).eps * sigma[0]
    result = dict(energy=total, top4_energy=float(energy[:4].sum()),
                  frobenius_norm=float(np.sqrt(total)),
                  numerical_rank=int(np.count_nonzero(sigma > tolerance)),
                  rank_tolerance=float(tolerance), zero_update=total == 0)
    for k in (1, 4, 8):
        fraction = float(energy[:k].sum() / total) if total else None
        if fraction is not None and not 0 <= fraction <= 1:
            raise ValueError(f"Top-{k} energy fraction outside [0, 1]")
        result[f"e{k}"] = fraction
    p = energy[energy > 0] / total if total else np.array([])
    residual = 1 - result["e4"] if total else None
    result.update(residual_beyond4=residual,
                  relative_rank4_error=float(np.sqrt(residual)) if total else None,
                  effective_rank=float(np.exp(-np.sum(p * np.log(p)))) if total else None)
    result.update({f"sigma_{i}": float(value) for i, value in enumerate(sigma, 1)})
    return result


def module_rows(state):
    expected = {
        f"model.layers.{layer}.{group}.{target}": (layer, target, out_dim, in_dim)
        for layer in range(LAYERS)
        for target, (group, out_dim, in_dim) in TARGETS.items()
    }
    keys = {f"{name}.lora_{factor}" for name in expected for factor in "AB"}
    if set(state) != keys:
        raise ValueError(f"Unexpected adapter coverage: {len(keys - set(state))} missing, "
                         f"{len(set(state) - keys)} extra factor keys")
    rows = []
    for name, (layer, target, out_dim, in_dim) in expected.items():
        a, b = (state[f"{name}.lora_{factor}"] for factor in "AB")
        if a.shape != (RANK, in_dim) or b.shape != (out_dim, RANK):
            raise ValueError(f"{name}: unexpected factor shapes {a.shape}, {b.shape}")
        rows.append(dict(module=name, layer=layer, target=target,
                         **spectrum(a.double().numpy(), b.double().numpy(), ALPHA / RANK)))
    return rows


def summarize(rows):
    nonzero = [row for row in rows if not row["zero_update"]]
    energy = sum(row["energy"] for row in rows)
    top4 = sum(row["top4_energy"] for row in rows)
    result = dict(modules=len(rows), nonzero_updates=len(nonzero),
                  zero_updates=len(rows) - len(nonzero), energy=energy, top4_energy=top4,
                  weighted_e4=top4 / energy if energy else None)
    for key in ("e4", "effective_rank"):
        values = np.array([row[key] for row in nonzero])
        for name, value in zip(("min", "p25", "median", "p75", "max"),
                               np.percentile(values, [0, 25, 50, 75, 100]) if values.size else [None]*5):
            result[f"{key}_{name}"] = float(value) if value is not None else None
        result[f"{key}_mean"] = float(values.mean()) if values.size else None
    for threshold in (90, 95, 99):
        count = sum(row["e4"] > threshold / 100 for row in nonzero)
        result[f"pct_nonzero_e4_above_{threshold}"] = 100 * count / len(nonzero) if nonzero else None
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_spectrum(path, rows):
    fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
    ranks = np.arange(1, RANK + 1)
    for target in TARGETS:
        curves = [[row[f"sigma_{i}"] / row["frobenius_norm"] for i in ranks]
                  for row in rows if row["target"] == target and not row["zero_update"]]
        if curves:
            ax.plot(ranks, np.median(curves, axis=0), marker=".", label=target)
    ax.axvline(4, linestyle="--", color="0.5", linewidth=1)
    ax.set(xlabel="Singular direction (descending order)",
           ylabel="Median singular value / update Frobenius norm",
           title="Learned LoRA-16 update spectra by projection", xticks=ranks)
    ax.legend(ncol=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Official state.pt or the trusted final-adapters tar archive")
    args = parser.parse_args()
    path = args.checkpoint.resolve()
    rows = module_rows(load_state(path))
    by_type = [dict(target=target, **summarize([r for r in rows if r["target"] == target]))
               for target in TARGETS]
    by_layer = [dict(layer=layer, **summarize([r for r in rows if r["layer"] == layer]))
                for layer in range(LAYERS)]
    overall = dict(checkpoint=str(path), checkpoint_member=MEMBER if tarfile.is_tarfile(path) else None,
                   checkpoint_sha256=SHA256, update=1250, rank=RANK, alpha=ALPHA,
                   precision="CPU float64", numpy_version=np.__version__,
                   torch_version=str(torch.__version__), **summarize(rows))
    output = Path(__file__).resolve().parent / "RQ2_lora_spectrum"
    output.mkdir(exist_ok=True)
    write_csv(output / "per_module_spectrum.csv", rows)
    write_csv(output / "by_module_type.csv", by_type)
    write_csv(output / "by_layer.csv", by_layer)
    (output / "overall_summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    plot_spectrum(output / "singular_value_spectrum.png", rows)
    (output / "appendix_methodology.md").write_text(
        f"""# LoRA-16 spectrum analysis method

- Input. Official LoRA-16 checkpoint at update 1,250. Source: `{path}`. Checkpoint SHA-256: `{SHA256}`.
- Coverage. All {len(rows)} adapters across {LAYERS} transformer layers and seven target projections.
- Calculation. Convert stored factors to CPU FP64. Form thin QR decompositions of B and A transposed. Take the SVD of the 16-by-16 core (alpha/r) R_B R_A transposed, using alpha=32 and r=16. Orthogonal outer factors preserve the singular values. Verify summed squared singular values against the factor Gram trace identity.
- Diagnostics. E4 is the share of squared singular values in the first four directions. Effective rank is the exponential of Shannon entropy of the normalized squared singular values. Numerical rank counts values strictly above max(input dimension, output dimension) times FP64 machine epsilon times the largest singular value. Fractions and entropy rank are undefined for zero updates and excluded from their summaries.
- Aggregation. Weighted E4 divides summed top-four energy by summed total energy. Unweighted quantiles and means describe individual modules. Effective-rank summaries describe module distributions. Threshold percentages use nonzero modules as the denominator and a strict greater-than comparison.
- Scope. The figure shows median per-module singular values normalized by each update's Frobenius norm, separately by projection. The analysis measures weight-update magnitude and does not establish the effect of directions 5–16 on held-out performance.
""", encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
