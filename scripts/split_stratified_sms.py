"""
Tam CSV'yi stratifiye train / test olarak ayırır (veri sızıntısız değerlendirme için).

Örnek (Kaggle spam.csv):
  python scripts/split_stratified_sms.py --input data/spam.csv --text-col v2 --label-col v1

Çıktı: data/sms_train.csv, data/sms_test.csv (varsayılan önek isimler)
Sonra:
  python train.py --data data/sms_train.csv --text-col v2 --label-col v1
  python evaluate.py --test-data data/sms_test.csv --text-col v2 --label-col v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _normalize_labels(series: pd.Series) -> pd.Series:
    """train.py ile aynı mantık (stratifiye için 0/1)."""
    s = series.copy()
    lower = s.astype(str).str.lower().str.strip()
    sms_map = {"ham": 0, "spam": 1}
    if lower.isin(sms_map.keys()).all():
        return lower.map(sms_map)
    return pd.to_numeric(s, errors="coerce")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def main() -> None:
    p = argparse.ArgumentParser(description="Stratifiye train/test CSV ayırma")
    p.add_argument("--input", required=True, type=str, help="Kaynak CSV")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--train-out", default="data/sms_train.csv")
    p.add_argument("--test-out", default="data/sms_test.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"Dosya yok: {inp}")

    df = _read_csv(inp)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"Sütun yok: {args.text_col}, {args.label_col}")

    y = _normalize_labels(df[args.label_col])
    ok = y.notna()
    if not ok.all():
        df = df.loc[ok].reset_index(drop=True)
        y = y.loc[ok].reset_index(drop=True)
    if y.isna().any() or len(y) == 0:
        raise SystemExit("Etiket sütunu okunamadı (ham/spam veya 0/1 beklenir).")

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    train_path = Path(args.train_out)
    test_path = Path(args.test_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"Train: {len(train_df)} satir -> {train_path}")
    print(f"Test:  {len(test_df)} satir -> {test_path}")
    print(f"text={args.text_col}, label={args.label_col}")


if __name__ == "__main__":
    main()
