"""
Kayıtlı model ile CSV metrikleri (ham=0, spam=1).

  python evaluate.py
  python evaluate.py --data data/sample.csv

Önerilen (veri sızıntısı yok):
  python scripts/split_stratified_sms.py --input data/spam.csv --text-col v2 --label-col v1
  python train.py --data data/sms_train.csv --text-col v2 --label-col v1
  python evaluate.py --test-data data/sms_test.csv --text-col v2 --label-col v1

Code Runner sistem Python kullanırsa, dosya .venv içindeki Python ile kendini yeniden çalıştırır.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def _running_as_evaluate_script() -> bool:
    """Code Runner / python dosya.py için True; 'import evaluate' için False."""
    if __name__ == "__main__":
        return True
    if sys.argv and len(sys.argv) > 0:
        try:
            return Path(sys.argv[0]).resolve().name.lower() == "evaluate.py"
        except OSError:
            return False
    return False


def _same_executable(a: Path, b: Path) -> bool:
    try:
        return a.resolve().as_posix().lower() == b.resolve().as_posix().lower()
    except OSError:
        return False


def _rerun_with_project_venv() -> None:
    """Sistem Python (ör. sklearn 1.7) ile açıldıysa, .venv ile tekrar çalıştır."""
    if not _running_as_evaluate_script():
        return
    if not _VENV_PYTHON.is_file():
        print(
            f"Uyarı: .venv bulunamadı ({_VENV_PYTHON}). "
            "Önce: python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt",
            file=sys.stderr,
        )
        return
    try:
        current = Path(sys.executable).resolve()
        wanted = _VENV_PYTHON.resolve()
    except OSError:
        return
    if _same_executable(current, wanted):
        return
    import subprocess

    script = Path(__file__).resolve()
    print(
        f"(evaluate.py) sklearn/model uyumu için proje sanal ortamı kullanılıyor:\n  {wanted}",
        file=sys.stderr,
    )
    completed = subprocess.run(
        [str(wanted), str(script), *sys.argv[1:]],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    code = completed.returncode if completed.returncode is not None else 1
    raise SystemExit(code)


_rerun_with_project_venv()

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from spam_detector.pipeline import SmsClassifierPipeline

# Model joblib, eğitimdeki sklearn sürümüyle uyumlu olmalı (ör. 1.8+).
_MIN_SKLEARN = (1, 8, 0)


def _sklearn_version_tuple() -> tuple[int, ...]:
    import sklearn

    parts: list[int] = []
    for p in sklearn.__version__.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)


def _ensure_sklearn_compatible() -> None:
    got = _sklearn_version_tuple()
    if got[:2] < _MIN_SKLEARN[:2]:
        msg = (
            f"\n>>> HATA: scikit-learn {'.'.join(map(str, got))} bulundu; model için {_MIN_SKLEARN[0]}.{_MIN_SKLEARN[1]}+ gerekli.\n"
            f">>> .venv oluştur:  python -m venv .venv\n"
            f">>> Kurulum:       .venv\\Scripts\\pip install -r requirements.txt\n"
        )
        print(msg, file=sys.stderr)
        raise SystemExit(1)


def _normalize_labels(series: pd.Series) -> pd.Series:
    s = series.copy()
    lower = s.astype(str).str.lower().str.strip()
    sms = {"ham": 0, "spam": 1}
    if lower.isin(sms.keys()).all():
        return lower.map(sms)
    legacy = {"real": 0, "true": 0, "fake": 1, "false": 1}
    if lower.isin(legacy.keys()).all():
        return lower.map(legacy)
    return pd.to_numeric(s, errors="coerce")


def _pick_default_csv() -> tuple[Path, str, str]:
    """Argümansız çalıştırma (Code Runner): sırayla dene."""
    candidates = [
        (ROOT / "data" / "spam.csv", "v2", "v1"),
        (ROOT / "data" / "sms_uci.csv", "text", "label"),
        (ROOT / "data" / "sample.csv", "text", "label"),
    ]
    for path, tc, lc in candidates:
        if path.is_file():
            return path, tc, lc
    raise SystemExit(
        "CSV bulunamadı. data/spam.csv, data/sms_uci.csv veya data/sample.csv ekleyin\n"
        "veya: python evaluate.py --data YOL.csv --text-col ... --label-col ..."
    )


def _resolve_csv_path(path_str: str) -> Path:
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _load_metrics_json(models_dir: Path) -> dict | None:
    mp = models_dir / "metrics.json"
    if not mp.is_file():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _paths_same_file(a: Path, b: str | None) -> bool:
    if not b:
        return False
    try:
        return a.resolve().as_posix().lower() == Path(b).resolve().as_posix().lower()
    except OSError:
        return False


def _pick_holdout_metrics(blob: dict, pipe: SmsClassifierPipeline) -> dict | None:
    """metrics.json içinden bu modele uygun tutulmuş test metrikleri."""
    mt = pipe.model_type
    if "Random Forest" in mt and isinstance(blob.get("rf"), dict):
        return blob["rf"]
    if "Logistic" in mt and isinstance(blob.get("logreg"), dict):
        return blob["logreg"]
    if isinstance(blob.get("logreg"), dict):
        return blob["logreg"]
    if isinstance(blob.get("rf"), dict):
        return blob["rf"]
    return None


def _print_training_reference(blob: dict | None, pipe: SmsClassifierPipeline) -> None:
    if not blob:
        print("(models/metrics.json yok — eğitim sonrası train.py oluşturur.)\n")
        return
    proto = blob.get("evaluation_protocol")
    td = blob.get("training_data")
    print("\n--- Eğitim kaydı (models/metrics.json) ---")
    if proto and proto.get("note_tr"):
        print(proto["note_tr"])
    if td:
        if td.get("path"):
            print(f"Eğitim CSV: {td['path']}")
        print(f"Eğitim sütunları: text={td.get('text_col')}, label={td.get('label_col')}")
        print(f"Temizlik sonrası örnek: {td.get('n_samples_after_clean')} | sınıf: {td.get('class_counts_0_ham_1_spam')}")
    hm = _pick_holdout_metrics(blob, pipe)
    if hm:
        print("\nTutulmuş test seti (eğitimde %20 ayrılan) metrikleri:")
        print(f"  Accuracy:  {hm['accuracy']:.4f}")
        print(f"  Precision: {hm['precision']:.4f}")
        print(f"  Recall:    {hm['recall']:.4f}")
        print(f"  F1:        {hm['f1']:.4f}")
    print("---\n")


def _warn_full_csv_if_training_file(
    data_path: Path,
    text_col: str,
    label_col: str,
    blob: dict | None,
    used_holdout_csv: bool,
) -> None:
    if used_holdout_csv or not blob:
        return
    td = blob.get("training_data") or {}
    path_s = td.get("path")
    if not path_s or not _paths_same_file(data_path, path_s):
        return
    if td.get("text_col") != text_col or td.get("label_col") != label_col:
        return
    print("=" * 62)
    print("UYARI: Bu CSV, eğitimde kullanılan dosya ile aynı görünüyor.")
    print("Aşağıdaki metrikler TÜM satırlar üzerinde; model çoğunu eğitimde görmüş olabilir.")
    print("Rapor için: scripts/split_stratified_sms.py → train (train.csv) → evaluate (--test-data test.csv)")
    print("=" * 62 + "\n")


def main() -> None:
    _ensure_sklearn_compatible()

    p = argparse.ArgumentParser(description="SMS spam modeli değerlendirme")
    p.add_argument(
        "--data",
        default="",
        help="CSV yolu (boş: data/spam.csv → sms_uci → sample). --test-data ile birlikte kullanma.",
    )
    p.add_argument(
        "--test-data",
        default="",
        help="Sadece tutulmuş test CSV (eğitimde kullanılmayan; önerilen protokol).",
    )
    p.add_argument("--text-col", default="text", help="Metin sütunu (spam.csv için genelde v2)")
    p.add_argument("--label-col", default="label", help="Etiket sütunu (spam.csv için genelde v1)")
    p.add_argument("--models-dir", default="models", help="Kayıtlı .joblib klasörü")
    p.add_argument("--artifact", default="pipeline_artifacts.joblib", help="Dosya adı")
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    metrics_blob = _load_metrics_json(models_dir)

    used_holdout = bool(str(args.test_data).strip())
    if used_holdout and str(args.data).strip():
        raise SystemExit("Ya --data ya da --test-data kullanın; ikisini birden vermeyin.")

    if used_holdout:
        data_path = _resolve_csv_path(str(args.test_data).strip())
        text_col = args.text_col
        label_col = args.label_col
        if text_col == "text" and label_col == "label" and data_path.name.lower() == "spam.csv":
            text_col, label_col = "v2", "v1"
        try:
            _show = data_path.relative_to(ROOT)
        except ValueError:
            _show = data_path
        print(f"(tutulmuş test) {_show}")
        print("Bu dosya eğitimde kullanılmamış olmalı (train/test ayrımına uy).")
    elif not str(args.data).strip():
        data_path, text_col, label_col = _pick_default_csv()
        print(f"(otomatik) Veri: {data_path.relative_to(ROOT)}  sütunlar: text={text_col}, label={label_col}")
    else:
        data_path = _resolve_csv_path(str(args.data).strip())
        text_col = args.text_col
        label_col = args.label_col
        if text_col == "text" and label_col == "label" and data_path.name.lower() == "spam.csv":
            text_col, label_col = "v2", "v1"
            print(f"(otomatik) Kaggle spam.csv için sütunlar: {text_col}, {label_col}")

    if not data_path.is_file():
        raise SystemExit(f"CSV bulunamadı: {data_path}")

    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="latin-1")
    y = _normalize_labels(df[label_col]).dropna().astype(int)
    X_raw = df.loc[y.index, text_col].astype(str)

    pipe = SmsClassifierPipeline.load(str(models_dir), args.artifact)
    _print_training_reference(metrics_blob, pipe)
    _warn_full_csv_if_training_file(data_path, text_col, label_col, metrics_blob, used_holdout)

    n0, n1 = pipe.label_names[0], pipe.label_names[1]

    truth: list[int] = []
    preds: list[int] = []
    for raw, yi in zip(X_raw.astype(str), y):
        label, _ = pipe.predict(raw)
        if label.startswith("Belirsiz"):
            continue
        pred = 1 if label == n1 else 0
        preds.append(pred)
        truth.append(int(yi))
    y_true = np.array(truth)
    y_pred = np.array(preds)

    print(f"Model: {pipe.model_type} ({args.artifact})")
    label = "Bu CSV üzerindeki tahminler" if not used_holdout else "Tutulmuş test CSV üzerindeki tahminler"
    print(f"{label} — örnek sayısı: {len(y_true)}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, target_names=[f"{n0}(0)", f"{n1}(1)"]))


if __name__ == "__main__":
    main()
