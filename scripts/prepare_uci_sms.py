"""
UCI Machine Learning Repository — SMS Spam Collection dosyasını CSV'ye çevirir.

İndirme (resmi kaynak):
  https://archive.ics.uci.edu/dataset/228/sms+spam+collection

ZIP içinden `SMSSpamCollection` dosyasını çıkarıp kullanın.

Kullanım:
  python scripts/prepare_uci_sms.py path/to/SMSSpamCollection -o data/sms_uci.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="UCI SMSSpamCollection -> CSV (text, label)")
    p.add_argument("input_path", help="SMSSpamCollection dosyasının yolu")
    p.add_argument("-o", "--out", default="data/sms_uci.csv", help="Çıktı CSV")
    args = p.parse_args()

    src = Path(args.input_path)
    if not src.is_file():
        raise SystemExit(f"Dosya bulunamadı: {src}")

    rows: list[tuple[str, int]] = []
    with open(src, encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            tag, msg = parts[0].strip().lower(), parts[1].strip()
            label = 1 if tag == "spam" else 0
            rows.append((msg, label))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for msg, label in rows:
            w.writerow([msg, label])

    print(f"Yazıldı: {out.resolve()} ({len(rows)} satır, 0=ham 1=spam)")


if __name__ == "__main__":
    main()
