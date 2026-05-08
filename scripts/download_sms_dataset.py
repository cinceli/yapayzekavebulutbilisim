"""
UCI sitesi 502/timeout verdiğinde: aynı SMS Spam verisinin güvenilir bir GitHub aynasından indirir.

Kaynak: PyCon 2016 tutorial verisi (UCI SMS Spam Collection ile aynı ham/spam içerik, TSV).
Orijinal atıf: Tiago et al., UCI ML Repository.

Kullanım (proje kökünden):
  python scripts/download_sms_dataset.py
  python scripts/download_sms_dataset.py -o data/sms_uci.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.request import urlopen

# UCI 502 olduğunda kullanılan ayna (aynı tip: ham/spam + TAB + mesaj)
DEFAULT_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
)


def main() -> None:
    p = argparse.ArgumentParser(description="SMS spam verisini indir -> CSV (text, label)")
    p.add_argument("-o", "--out", default="data/sms_uci.csv", help="Çıktı CSV")
    p.add_argument("--url", default=DEFAULT_URL, help="TSV kaynak URL")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"İndiriliyor: {args.url}")
    with urlopen(args.url, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    rows: list[tuple[str, int]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        tag, msg = parts[0].strip().lower(), parts[1].strip()
        label = 1 if tag == "spam" else 0
        rows.append((msg, label))

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for msg, label in rows:
            w.writerow([msg, label])

    print(f"Tamam: {out.resolve()} ({len(rows)} satır, 0=ham 1=spam)")


if __name__ == "__main__":
    main()
