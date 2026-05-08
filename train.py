"""
SMS spam: TF-IDF + Logistic Regression ve/veya Random Forest (SVD).

Çalıştırma (proje kök klasöründen):
  python train.py --demo
  python train.py --data data/sms_uci.csv --text-col text --label-col label
  python scripts/prepare_uci_sms.py path/to/SMSSpamCollection -o data/sms_uci.csv

Çıktı: models/ klasörü (.joblib + metrics.json)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from spam_detector.preprocessing import TextPreprocessor
from spam_detector.training import (
    DEFAULT_LABEL_NAMES,
    print_report,
    train_logistic_regression,
    train_random_forest_svd,
    train_result_to_joblib_dict,
)


def _normalize_labels(series: pd.Series) -> pd.Series:
    """Ham/spam (veya real/fake) metinleri 0/1 yapar; sayısal etiketleri olduğu gibi okur."""
    s = series.copy()
    lower = s.astype(str).str.lower().str.strip()
    sms_map = {"ham": 0, "spam": 1}
    if lower.isin(sms_map.keys()).all():
        return lower.map(sms_map)
    legacy = {
        "real": 0,
        "true": 0,
        "reliable": 0,
        "genuine": 0,
        "fake": 1,
        "false": 1,
    }
    if lower.isin(legacy.keys()).all():
        return lower.map(legacy)
    return pd.to_numeric(s, errors="coerce")


def _read_csv(path: Path) -> pd.DataFrame:
    """UTF-8; olmazsa latin-1 (Kaggle / eski SMS CSV için)."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_dataset(path: Path, text_col: str, label_col: str) -> tuple[pd.Series, pd.Series]:
    df = _read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV'de '{text_col}' ve '{label_col}' sütunları olmalı.")
    y = _normalize_labels(df[label_col]).dropna().astype(int)
    X = df.loc[y.index, text_col].astype(str)
    return X, y


def build_demo_data() -> tuple[pd.Series, pd.Series]:
    ham = [
        "Hey are we still meeting at 6 tonight?",
        "Thanks for lunch see you tomorrow",
        "Can you pick up milk on the way home",
        "Running 10 min late sorry",
        "Happy birthday hope you have a great day",
    ] * 12
    spam = [
        "WINNER claim your free prize now text YES to 80000",
        "URGENT you have won 1000GBP call 09061213251",
        "Free entry in weekly lottery text WIN to 8007 now",
        "You have 1 new voicemail dial 123 now premium rate",
        "Buy cheap meds online click this link now",
    ] * 12
    X = pd.Series(ham + spam)
    y = pd.Series([0] * len(ham) + [1] * len(spam))
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="SMS spam (ham/spam) model eğitimi")
    parser.add_argument("--data", type=str, default="", help="CSV dosya yolu")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--out", type=str, default="models", help="Eğitilmiş model klasörü")
    parser.add_argument("--demo", action="store_true", help="Sentetik SMS ile hızlı test")
    parser.add_argument(
        "--model",
        choices=("logreg", "rf", "both"),
        default="logreg",
        help="logreg | rf | both (ikisi + en iyi F1 varsayılan dosya)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_names = DEFAULT_LABEL_NAMES

    if args.demo:
        X_raw, y = build_demo_data()
    else:
        if not args.data:
            raise SystemExit("CSV yolu: --data ... veya --demo")
        X_raw, y = load_dataset(Path(args.data), args.text_col, args.label_col)

    pre = TextPreprocessor()
    X_clean = X_raw.map(pre.clean)
    mask = X_clean.str.len() > 0
    X_clean, y = X_clean[mask], y[mask]

    if len(y.unique()) < 2:
        raise SystemExit("En az iki sınıf (0=ham, 1=spam) gerekli.")

    def _metrics_common() -> dict:
        """Rapor / evaluate.py için: tutulmuş test protokolü ve veri özeti."""
        dist = {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()}
        if args.demo:
            data_path = None
            text_col = label_col = "(demo)"
        else:
            data_path = str(Path(args.data).resolve())
            text_col, label_col = args.text_col, args.label_col
        return {
            "evaluation_protocol": {
                "type": "stratified_holdout",
                "test_size": 0.2,
                "random_state": 42,
                "note_tr": (
                    "Model satır metrikleri (accuracy, precision, recall, F1), eğitimde "
                    "verinin %20'sine ayrılan tutulmuş test seti üzerindedir. "
                    "Tüm eğitim CSV'si üzerinde yeniden ölçüm yapmak genelde daha iyimser sonuç verir (veri sızıntısı)."
                ),
            },
            "training_data": {
                "path": data_path,
                "text_col": text_col,
                "label_col": label_col,
                "n_samples_after_clean": int(len(y)),
                "class_counts_0_ham_1_spam": dist,
            },
        }

    def dump_metrics(payload: dict) -> None:
        merged = {**_metrics_common(), **payload}
        (out_dir / "metrics.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"Metrikler: {out_dir / 'metrics.json'}")

    if args.model == "logreg":
        res = train_logistic_regression(pre, X_clean, y, label_names=label_names)
        print_report(res)
        joblib.dump(train_result_to_joblib_dict(res), out_dir / "pipeline_artifacts.joblib")
        print(f"Kaydedildi: {out_dir / 'pipeline_artifacts.joblib'}")
        dump_metrics({"logreg": res.metrics, "default_artifact": "pipeline_artifacts.joblib"})

    elif args.model == "rf":
        res = train_random_forest_svd(pre, X_clean, y, label_names=label_names)
        print_report(res)
        joblib.dump(train_result_to_joblib_dict(res), out_dir / "pipeline_artifacts.joblib")
        print(f"Kaydedildi: {out_dir / 'pipeline_artifacts.joblib'}")
        dump_metrics({"rf": res.metrics, "default_artifact": "pipeline_artifacts.joblib"})

    else:
        res_lr = train_logistic_regression(pre, X_clean, y, label_names=label_names)
        print_report(res_lr)
        joblib.dump(train_result_to_joblib_dict(res_lr), out_dir / "pipeline_logreg.joblib")
        print(f"Kaydedildi: {out_dir / 'pipeline_logreg.joblib'}\n")

        res_rf = train_random_forest_svd(pre, X_clean, y, label_names=label_names)
        print_report(res_rf)
        joblib.dump(train_result_to_joblib_dict(res_rf), out_dir / "pipeline_rf.joblib")
        print(f"Kaydedildi: {out_dir / 'pipeline_rf.joblib'}\n")

        if res_lr.metrics["f1"] >= res_rf.metrics["f1"]:
            shutil.copyfile(out_dir / "pipeline_logreg.joblib", out_dir / "pipeline_artifacts.joblib")
            winner = "logreg"
        else:
            shutil.copyfile(out_dir / "pipeline_rf.joblib", out_dir / "pipeline_artifacts.joblib")
            winner = "rf"
        print(f"Varsayılan pipeline (en yüksek F1: {winner}): pipeline_artifacts.joblib")
        dump_metrics(
            {
                "logreg": res_lr.metrics,
                "rf": res_rf.metrics,
                "default_by_f1": winner,
                "default_artifact": "pipeline_artifacts.joblib",
            }
        )


if __name__ == "__main__":
    main()
