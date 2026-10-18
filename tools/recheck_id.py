"""Offline recalculation of the "ID before cash" rule from saved debug timelines (no re-inference).

Shows which videos would get no_id_check if `card_or_document` (an ID card is a card) also counts as
"document shown", so the rule can be calibrated on clean videos first, then checked on violations.
usage: python tools/recheck_id.py v6
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sessions_of(events, count0, duration):
    ss = [[0.0, None]] if count0 >= 2 else []
    for e in events:
        if e["kind"] == "enter" and e["before"] >= 1:
            ss.append([e["t"], None])
        elif e["kind"] == "exit" and ss and ss[-1][1] is None and e["after"] <= 1:
            ss[-1][1] = e["t"]
    return [(s, end if end is not None else duration) for s, end in ss]


def flags(hands, events, count0, duration, id_classes):
    out = []
    for s, end in sessions_of(events, count0, duration):
        if end - s < 6.0:
            continue
        win = [h for h in hands if h.get("present") and s - 1.0 <= h["t"] <= end]
        cash = [h for h in win if h.get("in_hands") == "banknotes"]
        if len(cash) < 2:
            continue
        if not any(h.get("in_hands") in id_classes and h["t"] <= cash[0]["t"] for h in win):
            out.append(cash[0]["t"])
    return out


def main() -> None:
    tag = sys.argv[1]
    labels = {r["canonical"]: int(r["violation"]) for r in csv.DictReader((ROOT / "data/labels.csv").open(encoding="utf-8"))}
    for name, classes in (("strict id_card", {"id_card"}), ("id_card + card_or_document", {"id_card", "card_or_document"})):
        hit_v, hit_c = [], []
        for f in sorted((ROOT / "runs" / f"eval_{tag}").glob("*.json"), key=lambda p: int(p.stem.split("-")[1])):
            res = json.loads(f.read_text(encoding="utf-8"))
            dbg = json.loads((ROOT / "runs" / res["job_id"] / "debug.json").read_text(encoding="utf-8"))
            fl = flags(dbg.get("hands", []), dbg.get("events", []), (dbg.get("count") or [0])[0],
                       res["video"]["duration_sec"], classes)
            if fl:
                (hit_v if labels[f.stem] else hit_c).append(f"{f.stem}@{fl[0]:.0f}")
        print(f"{name:28s} violation videos flagged: {len(hit_v):2d} {hit_v}")
        print(f"{'':28s} clean videos flagged:     {len(hit_c):2d} {hit_c}")


if __name__ == "__main__":
    main()
