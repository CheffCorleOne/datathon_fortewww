"""Build data/labels.csv from the organizers' open_labels.xlsx.

Normalizes video ids (Latin "X" vs Cyrillic "Х"), parses Russian timestamps
like "22 секунда" / "1минута 8 секунда" into seconds, and collapses byte-identical
duplicate videos so one physical file carries all of its labelled events.
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "open"
VIDEOS = DATA / "videos"
XLSX = DATA / "open_labels.xlsx"
OUT = ROOT / "data" / "labels.csv"


def norm_id(name: str) -> str:
    return name.strip().replace("Х", "X").replace("х", "x")


def parse_ts(text: str | None) -> float | None:
    if not text:
        return None
    text = str(text).lower()
    minutes = re.search(r"(\d+)\s*мин", text)
    seconds = re.search(r"(\d+)\s*сек", text)
    if not minutes and not seconds:
        return None
    return (int(minutes.group(1)) * 60 if minutes else 0) + (int(seconds.group(1)) if seconds else 0)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    files = {norm_id(p.stem): p for p in VIDEOS.glob("*.mov")}
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb.active
    rows = []
    for name, ts, label in ws.iter_rows(min_row=2, values_only=True):
        if not name:
            continue
        vid = norm_id(str(name))
        path = files.get(vid)
        if path is None:
            raise SystemExit(f"label row {name!r} has no matching video")
        rows.append({
            "id": vid,
            "file": path.name,
            "violation": int(str(label).strip().lower().startswith("наруш")),
            "t_sec": parse_ts(ts),
        })

    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        r["hash"] = sha(VIDEOS / r["file"])
        by_hash[r["hash"]].append(r)
    for group in by_hash.values():
        canonical = min(group, key=lambda r: (len(r["id"]), r["id"]))["id"]
        for r in group:
            r["canonical"] = canonical

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "canonical", "file", "violation", "t_sec", "hash"])
        w.writeheader()
        w.writerows(rows)

    uniq = {r["canonical"] for r in rows}
    pos = {r["canonical"] for r in rows if r["violation"]}
    print(f"{len(rows)} label rows, {len(uniq)} unique videos, {len(pos)} with violations -> {OUT}")


if __name__ == "__main__":
    main()
