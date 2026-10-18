"""Recompute open-set metrics from an eval CSV with some event types switched off (no re-inference).

usage: python tools/rescore.py full2 --drop phone_use count_without_client [--min-conf 0.6]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--type-min", nargs="*", default=[], help="type=min_conf pairs, e.g. manual_count=0.9")
    ap.add_argument("--drop-source", nargs="*", default=[], help="drop events whose detected_by contains this")
    args = ap.parse_args()
    type_min = {k: float(v) for k, v in (x.split("=") for x in args.type_min)}
    rows = list(csv.DictReader((ROOT / "runs" / f"eval_{args.tag}.csv").open(encoding="utf-8")))
    tp = fp = fn = tn = 0
    hits, per_type_fp, per_type_tp = [], Counter(), Counter()
    for r in rows:
        gt = int(r["gt"])
        times = json.loads(r["gt_times"])
        evs = [e for e in json.loads(r["events"])
               if e["type_code"] not in args.drop and (e.get("confidence") or 0) >= args.min_conf
               and (e.get("confidence") or 0) >= type_min.get(e["type_code"], 0.0)
               and not any(s in (e.get("detected_by") or "") for s in args.drop_source)]
        pred = bool(evs)
        for e in evs:
            (per_type_tp if gt else per_type_fp)[e["type_code"]] += 1
        if gt and pred:
            tp += 1
            for t in times:
                hits.append(min(abs(e["timecode_sec"] - t) for e in evs))
        elif gt:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
        mark = "OK " if bool(gt) == pred else "ERR"
        print(f"{mark} {r['id']:5s} gt={'V' if gt else '-'} {times!s:12s} -> "
              + "; ".join(f"{e['timecode_sec']:.1f} {e['type_code']}" for e in evs))
    n = tp + fp + fn + tn
    print(f"\nvideos={n} acc={(tp + tn) / max(1, n):.3f} TP={tp} FN={fn} FP={fp} TN={tn}")
    if hits:
        print(f"timecode: within {args.tol}s {sum(h <= args.tol for h in hits)}/{len(hits)}, "
              f"median {sorted(hits)[len(hits) // 2]:.1f}s")
    print("events in violation videos:", dict(per_type_tp))
    print("events in clean videos:    ", dict(per_type_fp))


if __name__ == "__main__":
    main()
