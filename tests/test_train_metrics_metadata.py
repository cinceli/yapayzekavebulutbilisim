"""train.py --demo çıktısında metrics.json değerlendirme protokolü alanları olsun."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_demo_train_writes_evaluation_protocol(tmp_path) -> None:
    out = tmp_path / "models_out"
    r = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--demo", "--out", str(out)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    blob = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert blob.get("evaluation_protocol", {}).get("type") == "stratified_holdout"
    assert "training_data" in blob
    assert blob["training_data"].get("n_samples_after_clean", 0) > 0
