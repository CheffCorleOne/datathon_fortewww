"""Evaluate the pipeline on the labelled open set.

Default: calls the pipeline in-process (fast iteration). With --api URL it goes through the
real HTTP API exactly like the jury (POST /api/v1/analyze, poll GET /api/v1/jobs/{id}).

Writes runs/eval_<tag>.csv and prints verdict accuracy, misses, false alarms, timecode error.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEOS = ROOT / "data" / "open" / "videos"
LABELS = ROOT / "data" / "labels.csv"


def load_labels() -> dict[str, dict]:
    """canonical id -> {file, violation, times[]} (duplicates merged)."""
    out: dict[str, dict] = {}
    for r in csv.DictReader(LABELS.open(encoding="utf-8")):
        c = out.setdefault(r["canonical"], {"file": None, "violation": 0, "times": []})
        if r["id"] == r["canonical"]:
            c["file"] = r["file"]
        c["violation"] |= int(r["violation"])
        if r["t_sec"]:
            c["times"].append(float(r["t_sec"]))
    return out


def run_local(path: Path) -> dict:
    from app.pipeline.analyze import analyze_video
    return analyze_video(path, lambda *_: None)


def run_api(path: Path, url: str, key: str) -> dict:
    import httpx
    with httpx.Client(base_url=url, headers={"X-API-Key": key}, timeout=120) as c:
        with path.open("rb") as f:
            r = c.post("/api/v1/analyze", files={"file": (path.name, f, "video/quicktime")})
        r.raise_for_status()
        job = r.json()["job_id"]
        t0 = time.time()
        while time.time() - t0 < 600:
            j = c.get(f"/api/v1/jobs/{job}").json()
            if j["status"] in ("done", "failed"):
                return j
            time.sleep(2)
    raise TimeoutError(job)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--api")
    ap.add_argument("--key", default="")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--tol", type=float, default=3.0, help="timecode tolerance for a hit, seconds")
    args = ap.parse_args()

    labels = load_labels()
    rows = []
    tp = fp = fn = tn = 0
    dt_err = []
    type_counts = defaultdict(int)
    for vid, lab in sorted(labels.items(), key=lambda kv: int(kv[0].split("-")[1])):
        if args.only and vid not in args.only:
            continue
        path = VIDEOS / lab["file"]
        t0 = time.time()
        res = run_api(path, args.api, args.key) if args.api else run_local(path)
        took = time.time() - t0
        dump = ROOT / "runs" / f"eval_{args.tag}"
        dump.mkdir(parents=True, exist_ok=True)
        slim = json.loads(json.dumps(res, default=str))
        for v in slim.get("violations", []):
            for f in v.get("evidence_frames", []):
                f["image_base64"] = f"<{len(f['image_base64'])} chars>"
        (dump / f"{vid}.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
        pred = res.get("verdict") == "violation"
        evs = res.get("violations", [])
        for e in evs:
            type_counts[e.get("type_code", e["type"])] += 1
        hits = []
        for t in lab["times"]:
            best = min((abs(e["timecode_sec"] - t) for e in evs), default=None)
            hits.append(best)
            if best is not None:
                dt_err.append(best)
        if lab["violation"] and pred:
            tp += 1
        elif lab["violation"]:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
        mark = "OK " if bool(lab["violation"]) == pred else "ERR"
        ev_txt = "; ".join(f"{e['timecode_sec']:.1f}s {e.get('type_code', e['type'])} ({e.get('confidence', 0):.2f})" for e in evs)
        print(f"{mark} {vid:6s} gt={'V' if lab['violation'] else '-'} {lab['times']!s:14s} "
              f"pred={'V' if pred else '-'} dt={[round(h, 1) if h is not None else None for h in hits]} "
              f"[{took:5.1f}s] {ev_txt}", flush=True)
        rows.append({"id": vid, "gt": lab["violation"], "gt_times": json.dumps(lab["times"]), "pred": int(pred),
                     "events": json.dumps([{k: e.get(k) for k in ("timecode_sec", "type_code", "confidence", "detected_by")}
                                           for e in evs]),
                     "dt": json.dumps(hits), "sec": round(took, 1),
                     "status": res.get("status", "done")})

    n = tp + fp + fn + tn
    print(f"\nvideos={n} acc={(tp + tn) / max(n, 1):.3f} TP={tp} FN={fn} FP={fp} TN={tn}")
    if dt_err:
        within = sum(d <= args.tol for d in dt_err)
        print(f"timecode |dt|: median={sorted(dt_err)[len(dt_err) // 2]:.1f}s, within {args.tol}s: {within}/{len(dt_err)}")
    print("event types:", dict(type_counts))
    out = ROOT / "runs" / f"eval_{args.tag}.csv"
    out.parent.mkdir(exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("->", out)


if __name__ == "__main__":
    main()
