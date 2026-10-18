"""Pretty-print the debug timeline saved by tools/eval.py for one video."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tag, vid = sys.argv[1], sys.argv[2]
    d = json.loads((ROOT / "runs" / f"eval_{tag}" / f"{vid}.json").read_text(encoding="utf-8"))
    dbg = d.get("debug")
    if dbg is None:  # result came through the API: debug lives next to the job
        dbg = json.loads((ROOT / "runs" / d["job_id"] / "debug.json").read_text(encoding="utf-8"))
    print("landmarks:", dbg.get("landmarks"))
    act = dbg.get("machine_act")
    if act:
        import numpy as np
        a = np.asarray(act)
        base = float(np.median(a)) + 1e-3
        hot = [i / 4 for i, v in enumerate(a) if v > max(4 * base, 6.0)]
        print(f"machine act median={base:.1f} hot_t={hot}")
    print("events:", [(e["kind"], e["before"], e["after"], round(e["t"], 1)) for e in dbg.get("events", [])])
    for h in dbg.get("hands", []):
        if h.get("present"):
            print(f"  {h['t']:5.1f} {h.get('in_hands', '?'):17s} flip={h.get('flipping_through_notes')}")
    for c in dbg.get("checks", []):
        print("  check", c)
    for v in d.get("violations", []):
        print("=>", v["timecode_sec"], v.get("type_code"), v.get("confidence"))


if __name__ == "__main__":
    main()
