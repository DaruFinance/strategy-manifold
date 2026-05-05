"""Smoke test for the analysis machinery itself.

Runs the script with `--synthetic` (three Gaussian blobs in 12-D) so the
embedding + connectivity machinery is exercised deterministically without
touching the real-data Parquet substrate. End-to-end real-data runs go
through the cross-asset orchestrator.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_demo_runs():
    res = subprocess.run(
        [sys.executable, "scripts/manifold.py", "--synthetic"],
        cwd=ROOT, capture_output=True, text=True, timeout=240,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads((ROOT / "manifold.json").read_text())
    assert out["mode"] == "synthetic"
    assert out["n"] == 3000
    # Synthetic demo: well-separated blobs => modularity should be very high
    mp_umap = out["connectivity_umap"]["modularity_proxy"]
    assert mp_umap > 0.95, f"modularity_umap={mp_umap}"
    for fname in ["fig_umap_synthetic.png", "fig_pca_synthetic.png"]:
        assert (ROOT / "figures" / fname).is_file()


if __name__ == "__main__":
    test_synthetic_demo_runs()
    print("OK")
