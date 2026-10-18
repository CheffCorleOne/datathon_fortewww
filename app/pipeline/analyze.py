"""End-to-end analysis of one video.

1. decode at 4 fps;
2. RT-DETR detections -> in-room people timeline, enter/exit/empty events, workstation (laptop) box;
3. local VLM on cropped regions:
   - "hands scan": every ~1.5 s, 3 frames of the cashier -> manual counting / phone / eating;
   - event checks: when the room empties (laptop locked? items left on desk?), when someone enters
     (cash lying on the desk? third person?), when the cashier arrives (door left open?),
     when the client leaves (cash still being handled?);
4. temporal rules -> events with timecode, evidence frames and a normative reference.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import cv2
import numpy as np

from .. import config
from . import scene as scene_mod
from . import video, vlm
from .detect import Detector
from .report import Event, build

log = logging.getLogger("analyze")
FPS = 4.0

HANDS_SCHEMA = {
    "type": "object",
    "properties": {
        "in_hands": {"type": "string",
                     "enum": ["banknotes", "id_card", "phone", "card_or_document", "cup_or_food", "nothing", "other"]},
        "flipping_through_notes": {"type": "boolean"},
    },
    "required": ["in_hands", "flipping_through_notes"],
}
HANDS_PROMPT = (
    "These are {n} consecutive video frames (0.5 s apart) of a bank cashier. What is in the cashier's hands? "
    "Note: tenge banknotes are rectangular paper notes in blue, green, brown or pink colours, often held as a "
    "small fan or stack; id_card = a single identity card with a small face photo, or a passport; a phone is a "
    "single thick dark/black slab with a screen. Reply in JSON.")

LAPTOP_SCHEMA = {"type": "object", "properties": {"screen": {"type": "string", "enum": ["lock", "off", "desktop", "app"]}},
                 "required": ["screen"]}
LAPTOP_PROMPT = ("Close-up of a laptop screen. Classify what the screen shows: 'lock' = Windows lock screen (big "
                 "clock/date over a photo, or a login box), 'off' = black/dark screen, 'desktop' = Windows desktop "
                 "with taskbar or icons, 'app' = an open program window (table, document, browser). Reply in JSON.")

GROUND_SCHEMA = {"type": "object", "properties": {
    "counting_machine": {"type": "array", "items": {"type": "integer"}},
    "safe": {"type": "array", "items": {"type": "integer"}},
    "door": {"type": "array", "items": {"type": "integer"}}},
    "required": ["counting_machine", "safe", "door"]}
GROUND_PROMPT = ("Locate objects in this bank cash room image and give bounding boxes as [x1, y1, x2, y2] in "
                 "relative coordinates 0-1000: counting_machine = the black banknote counting machine on the desk; "
                 "safe = the grey metal safe cabinet; door = the glass entrance door. Reply in JSON.")

DESK_SCHEMA = {"type": "object", "properties": {
    "items_left": {"type": "boolean"}, "banknotes_on_desk": {"type": "boolean"},
    "items": {"type": "array", "items": {"type": "string"}}},
    "required": ["items_left", "banknotes_on_desk", "items"]}
DESK_PROMPT = ("The cashier has left her workplace. Look at the top surface of the desk. Besides the laptop, the "
               "desk telephone and the black banknote counting machine, is anything left lying on the desk: "
               "banknotes/cash, a bag or sack, a tablet, cups, documents? Reply in JSON.")

CASH_SCHEMA = {"type": "object", "properties": {"banknotes_lying_on_desk": {"type": "boolean"}},
               "required": ["banknotes_lying_on_desk"]}
CASH_PROMPT = ("Look at the top surface of the desk in this bank cash room. Are paper banknotes or a bundle of "
               "cash lying on the desk (not held in anyone's hands)? Reply in JSON.")

DOOR_SCHEMA = {"type": "object", "properties": {"door": {"type": "string", "enum": ["closed", "open"]}},
               "required": ["door"]}
DOOR_PROMPT = ("This is the entrance of a bank cash room: a glass door in a glass wall. Is the glass door fully "
               "closed, or is it left open / ajar? Reply in JSON.")

BAG_SCHEMA = {"type": "object", "properties": {"personal_items_on_desk": {"type": "boolean"}},
              "required": ["personal_items_on_desk"]}
BAG_PROMPT = ("Are personal belongings of the cashier (handbag, purse, backpack, coat, jacket, clothes) lying on "
              "the desk? A green cash-collection sack is NOT a personal belonging. Reply in JSON.")

THIRD_SCHEMA = {"type": "object", "properties": {"person_on_cashier_side": {"type": "boolean"}},
                "required": ["person_on_cashier_side"]}
THIRD_PROMPT = ("Bank cash room. A person has just entered while the cashier is working. Is this newly entered "
                "person standing behind or right next to the cashier on the employee side of the desk, or handing "
                "something to the cashier while another client is already being served? Reply in JSON.")

_warm = False
# Detectors with no positive training examples and many false alarms on the open set stay off.
ENABLE_PHONE = False
ENABLE_CLIENT_LEFT = True   # strict version: client gone >= 5 s, VLM and hands scan both see cash work
ENABLE_ID_CHECK = False     # ID before cash (п. 19 пп. 1): on the open set 0 extra hits, 5 false alarms -> off
ENABLE_SR = True            # Swin2SR x2 for the held-object and laptop close-ups
ENABLE_POSE = True          # ViTPose wrists vs counting-machine zone around manual-count candidates
MANUAL_MIN_WINDOWS = 2  # 2-window runs get confidence 0.74 (3+: >= 0.86); final cut via MANUAL_MIN_CONF
MANUAL_MIN_CONF = 0.0


def warmup() -> None:
    global _warm
    try:
        Detector.get()
        if ENABLE_SR:
            from .enhance import SuperRes
            SuperRes.get()  # load once at startup, not inside the first jury request
        if ENABLE_POSE:
            from .pose import Pose
            Pose.get()
        if vlm.available():
            vlm.ask([np.zeros((64, 64, 3), np.uint8)], "Reply with the word ok.", num_predict=3)
        _warm = True
    except Exception:  # noqa: BLE001
        log.exception("warmup failed")


# ---------------------------------------------------------------- crops

def _crop(img: np.ndarray, box, pad: float = 0.2, side: int = 640, pad_y: float | None = None,
          enhance: bool = False) -> np.ndarray:
    """Cut `box` (+padding) and scale its long side to `side`.

    enhance=True routes the enlargement through the super-resolution middleware (enhance.py) instead of
    bicubic interpolation — used for the close-ups where fine detail decides the answer.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    py = pad if pad_y is None else pad_y
    x1, y1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - py * bh))
    x2, y2 = min(w, int(x2 + pad * bw)), min(h, int(y2 + py * bh))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img
    c = img[y1:y2, x1:x2]
    s = side / max(c.shape[:2])
    size = (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s)))
    if enhance and ENABLE_SR:
        from .enhance import upscale
        return upscale(c, size)
    return cv2.resize(c, size, interpolation=cv2.INTER_CUBIC)


def _union(boxes):
    b = np.asarray(boxes, dtype=float)
    return (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())


def _desk_box(sc: scene_mod.Scene):
    """Region around the workstation: laptop + counting machine, grown to cover the desk surface."""
    w, h = sc.frame_size
    parts = [b for b in (sc.laptop, sc.landmarks.get("machine")) if b is not None]
    if not parts:
        return (0, int(0.2 * h), w, int(0.75 * h))
    x1, y1, x2, y2 = _union(parts)
    bw, bh = x2 - x1, y2 - y1
    ref = max(bw, 0.25 * w)
    return (max(0, x1 - 0.5 * ref), max(0, y1 - 0.4 * bh), min(w, x2 + 0.5 * ref), min(h, y2 + 0.6 * ref))


def _ground(clip: video.Clip, sc: scene_mod.Scene) -> dict:
    """Ask the VLM once where the counting machine, safe and door are (relative 0-1000 boxes)."""
    i = int(np.argmin(sc.count)) if len(sc.count) else 0
    img = clip.frames[i]
    h, w = img.shape[:2]
    out: dict = {}
    try:
        r = vlm.ask([img], GROUND_PROMPT, schema=None, max_side=1024, num_predict=200)
        import json
        import re
        m = re.search(r"\{.*\}", str(r), re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception as exc:  # noqa: BLE001
        log.warning("grounding failed: %s", exc)
        return out
    for key, name in (("counting_machine", "machine"), ("safe", "safe"), ("door", "door")):
        b = data.get(key)
        while isinstance(b, list) and b and isinstance(b[0], list):
            b = b[0]
        if not (isinstance(b, list) and len(b) == 4):
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            continue
        out[name] = (x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000)
    return out


# ---------------------------------------------------------------- main

def analyze_video(path, progress: Callable[[str, float], None]) -> dict[str, Any]:
    t0 = time.time()
    clip = video.load(str(path), sample_fps=FPS, max_side=960, progress=progress)
    progress("detecting", 0.15)
    dets = Detector.get()(clip.frames, thr=0.3, progress=progress)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # give the VRAM back to the VLM between stages
    except Exception:  # noqa: BLE001
        pass
    progress("tracking", 0.5)
    sc = scene_mod.build(dets, clip.times, clip.size, FPS)
    use_vlm = vlm.available()
    events: list[Event] = []
    debug: dict[str, Any] = {"events": sc.events, "count": sc.count.tolist(), "vlm": use_vlm}
    if use_vlm:
        progress("verifying", 0.55)
        sc.landmarks = _ground(clip, sc)
        sc.machine_act = machine_activity(clip, sc)
        debug["landmarks"] = {k: [round(v, 1) for v in b] for k, b in sc.landmarks.items()}
        if sc.machine_act is not None:
            debug["machine_act"] = [round(float(v), 1) for v in sc.machine_act]
        hands = _hands_scan(clip, sc, progress, deadline=t0 + config.JOB_TIMEOUT_SEC * 0.6)
        debug["hands"] = hands
        debug["episode_checks"] = []
        events += _from_hands(clip, sc, hands, debug["episode_checks"], dets)
        progress("events", 0.85)
        events += _event_checks(clip, sc, dets, hands, debug)
    events = _dedupe(events)
    progress("report", 0.95)
    out = build(clip, events)
    out["debug"] = debug
    log.info("analyzed %s in %.1fs: %d events", path, time.time() - t0, len(events))
    return out


# ---------------------------------------------------------------- hands scan

def _hands_scan(clip: video.Clip, sc: scene_mod.Scene, progress, deadline: float) -> list[dict]:
    dur = clip.duration
    step = max(1.5, dur / 45.0)
    out: list[dict] = []
    t = 1.0
    windows = []
    while t < dur - 0.5:
        windows.append(t)
        t += step
    for k, tc in enumerate(windows):
        if time.time() > deadline:
            log.warning("hands scan cut at %.1fs (time budget)", tc)
            break
        ts = [tc - 0.5, tc, tc + 0.5]
        idx = [clip.index_at(x) for x in ts]
        boxes = [sc.cashier_box(i) for i in idx]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            out.append({"t": tc, "present": False})
            continue
        ub = _union(boxes)
        crops = [_crop(clip.frames[i], ub, pad=0.15, side=640, pad_y=0.1) for i in idx]
        try:
            r = vlm.ask(crops, HANDS_PROMPT.format(n=len(crops)), schema=HANDS_SCHEMA, max_side=640, num_predict=40)
        except Exception as exc:  # noqa: BLE001
            log.warning("hands vlm failed at %.1f: %s", tc, exc)
            continue
        r = dict(r) if isinstance(r, dict) else {}
        held = r.get("in_hands")
        r["banknotes_in_hands"] = held == "banknotes"
        r["counting_banknotes_by_hand"] = held == "banknotes" and bool(r.get("flipping_through_notes"))
        r["phone_in_hands"] = held == "phone"
        r["eating_or_drinking"] = held == "cup_or_food"
        r.update({"t": tc, "present": True, "box": [float(v) for v in ub]})
        out.append(r)
        progress("verifying", 0.55 + 0.3 * (k + 1) / max(1, len(windows)))
    return out


def _runs(hands: list[dict], pred) -> list[list[dict]]:
    runs, cur = [], []
    for h in hands:
        if h.get("present") and pred(h):
            cur.append(h)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _box_list(b):
    return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), "")] if b else []


def machine_activity(clip: video.Clip, sc: scene_mod.Scene) -> np.ndarray | None:
    """Per-frame motion energy inside the counting-machine box (hands/notes moving at the machine)."""
    box = sc.landmarks.get("machine")
    if box is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    h, w = clip.frames[0].shape[:2]
    x1, y1 = max(0, x1 - bw // 4), max(0, y1 - bh // 4)
    x2, y2 = min(w, x2 + bw // 4), min(h, y2 + bh // 4)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    prev = None
    energy = np.zeros(len(clip.frames))
    for i, f in enumerate(clip.frames):
        g = cv2.cvtColor(cv2.resize(f[y1:y2, x1:x2], (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if prev is not None:
            energy[i] = float(np.mean(np.abs(g - prev)))
        prev = g
    return energy


def _machine_used(act: np.ndarray | None, clip: video.Clip, t0: float, t1: float) -> bool:
    if act is None or len(act) < 3:
        return False
    base = float(np.median(act)) + 1e-3
    i0, i1 = clip.index_at(t0), clip.index_at(t1)
    seg = act[i0:i1 + 1]
    return bool(len(seg) and (np.sum(seg > max(4.0 * base, 6.0)) >= 2))


EPISODE_SCHEMA = {"type": "object", "properties": {
    "notes_put_into_machine": {"type": "boolean"},
    "notes_end_up": {"type": "string", "enum": ["counting_machine", "given_to_client", "safe_or_drawer",
                                                "kept_in_hands_or_desk", "unclear"]}},
    "required": ["notes_put_into_machine", "notes_end_up"]}
EPISODE_PROMPT = ("{n} frames in time order (about 1 s apart) of a bank cashier who first counts banknotes by "
                  "hand. The black box on the desk is the banknote counting machine. Does she then put the "
                  "banknotes into the counting machine? Where do the notes end up? Reply in JSON.")


def _machine_after(clip: video.Clip, sc: scene_mod.Scene, run: list[dict], checks: list) -> bool:
    """Episode-level VLM check: after the hand count, do the notes go through the counting machine?"""
    t0, t1 = run[0]["t"], run[-1]["t"]
    boxes = [h["box"] for h in run if h.get("box")]
    if sc.landmarks.get("machine") is not None:
        boxes.append(sc.landmarks["machine"])
    if not boxes:
        return False
    ub = _union(boxes)
    ts = list(np.arange(t0, min(clip.duration - 0.3, t1 + 6.0) + 1e-6, 1.2))[:6]
    crops = [_crop(clip.frames[clip.index_at(t)], ub, pad=0.1, side=448, pad_y=0.1) for t in ts]
    try:
        r = vlm.ask(crops, EPISODE_PROMPT.format(n=len(crops)), schema=EPISODE_SCHEMA, max_side=448, num_predict=40)
    except Exception as exc:  # noqa: BLE001 - a failed check must not drop the whole video
        log.warning("episode check failed: %s", exc)
        r = {}
    checks.append({"check": "machine_after_count", "t": t0, "r": r})
    return isinstance(r, dict) and bool(r.get("notes_put_into_machine"))


HELD_GROUND_PROMPT = ("This is a crop of a bank cashier. Locate the object she is holding in her hands. Reply in "
                      "JSON {\"bbox_2d\": [x1, y1, x2, y2]} with coordinates relative 0-1000 of this image.")
HELD_SCHEMA = {"type": "object", "properties": {"object": {"type": "string", "enum": [
    "banknotes", "id_card_or_passport", "phone_or_phone_case", "plastic_card", "document_or_receipt",
    "envelope_or_bag", "cup_or_food", "nothing", "other"]}}, "required": ["object"]}
HELD_PROMPT = ("Close-up crops (several moments) of the object in a bank cashier's hands. Classify it. "
               "banknotes = several thin flexible printed paper notes held as a stack or fan, coloured print with "
               "patterns, edges of multiple sheets visible, sheets bend. id_card_or_passport = one rigid card with a "
               "small photo of a face and text (identity card), or a passport booklet; a single item, not a stack. "
               "phone_or_phone_case = one thick rigid slab with rounded corners, may have a camera bump or a case. "
               "plastic_card = a single small rigid bank card without a face photo. document_or_receipt = white paper "
               "with text. Reply in JSON.")


def _held_object(clip: video.Clip, person_box, tc: float) -> tuple[str | None, tuple | None]:
    """Ground the object in the cashier's hands, then classify a tight close-up of it."""
    import re
    h, w = clip.frames[0].shape[:2]
    bw, bh = person_box[2] - person_box[0], person_box[3] - person_box[1]
    x0, y0 = max(0, person_box[0] - 0.1 * bw), max(0, person_box[1] - 0.05 * bh)
    x1, y1 = min(w, person_box[2] + 0.1 * bw), min(h, person_box[3] + 0.05 * bh)
    person = _crop(clip.frames[clip.index_at(tc)], person_box, pad=0.1, side=640, pad_y=0.05)
    obj_box = None
    try:
        raw = vlm.ask([person], HELD_GROUND_PROMPT, max_side=640, num_predict=40)
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(raw))][:4]
        if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
            cw, ch = x1 - x0, y1 - y0
            obj_box = (x0 + nums[0] / 1000 * cw, y0 + nums[1] / 1000 * ch,
                       x0 + nums[2] / 1000 * cw, y0 + nums[3] / 1000 * ch)
    except Exception as exc:  # noqa: BLE001
        log.warning("held-object grounding failed: %s", exc)
    box = obj_box or person_box
    pad = 0.6 if obj_box else 0.0
    crops = [_crop(clip.frames[clip.index_at(tc + d)], box, pad=pad, side=448, enhance=True) for d in (-0.5, 0.0, 0.5)]
    try:
        r = vlm.ask(crops, HELD_PROMPT, schema=HELD_SCHEMA, max_side=448, num_predict=20)
    except Exception as exc:  # noqa: BLE001
        log.warning("held-object check failed: %s", exc)
        return None, obj_box
    return (r.get("object") if isinstance(r, dict) else None), obj_box


def _machine_touches(clip: video.Clip, sc: scene_mod.Scene, s: float, e: float) -> dict:
    """Count frames where a cashier wrist is inside the (enlarged) counting-machine box:
    4 s before the episode, during it, and 6 s after. Empty dict when there is no machine or no pose."""
    machine = sc.landmarks.get("machine")
    if machine is None or not ENABLE_POSE:
        return {}
    x1, y1, x2, y2 = machine
    mw, mh = x2 - x1, y2 - y1
    zone = (x1 - 0.25 * mw, y1 - 0.25 * mh, x2 + 0.25 * mw, y2 + 0.25 * mh)
    idx, boxes = [], []
    for i, t in enumerate(clip.times):
        if s - 4.0 <= t <= e + 6.0:
            b = sc.cashier_box(i)
            if b is not None:
                idx.append(i)
                boxes.append(b)
    if not idx:
        return {}
    try:
        from .pose import Pose
        wr = Pose.get().wrists([clip.frames[i] for i in idx], boxes)
    except Exception as exc:  # noqa: BLE001 - pose is an extra signal, never a hard dependency
        log.warning("pose failed: %s", exc)
        return {}
    out = {"before": 0, "during": 0, "after": 0}
    for i, w in zip(idx, wr):
        if any(sc_ > 0.4 and zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3] for x, y, sc_ in w):
            t = clip.times[i]
            out["before" if t < s else "after" if t > e else "during"] += 1
    return out


def _from_hands(clip: video.Clip, sc: scene_mod.Scene, hands: list[dict], checks: list | None = None,
                dets: list | None = None) -> list[Event]:
    dets = dets or [[] for _ in clip.frames]
    ev: list[Event] = []
    checks = checks if checks is not None else []
    # manual counting not followed by the machine; tolerate one missed window inside a run
    flags = [bool(h.get("present") and h.get("counting_banknotes_by_hand")) for h in hands]
    for k in range(1, len(flags) - 1):
        if not flags[k] and flags[k - 1] and flags[k + 1]:
            flags[k] = True
    marked = [dict(h, _m=f) for h, f in zip(hands, flags)]
    for run in _runs(marked, lambda h: h["_m"]):
        t_start, t_end = run[0]["t"], run[-1]["t"]
        if len(run) < MANUAL_MIN_WINDOWS:
            continue  # 2-window runs were mostly notes taken out of the machine on the open set
        if _machine_after(clip, sc, run, checks):
            continue  # pre-count by hand, then the machine: normal practice
        mid = run[len(run) // 2]
        # Is it really money? Close-up of the held object at two moments; any "banknotes" keeps the run
        # (on the open set this removed phone/card/document runs without losing a single true violation).
        votes, obj_boxes = [], {}
        for h in (mid, run[max(0, len(run) // 2 - 1)]):
            if not h.get("box"):
                continue
            kind, ob = _held_object(clip, h["box"], h["t"])
            votes.append(kind)
            if ob is not None:
                obj_boxes[h["t"]] = ob
        checks.append({"check": "held_object", "t": mid["t"], "votes": votes})
        if votes and "banknotes" not in votes:
            continue
        # Hands + counting-machine zone (ViTPose wrists): notes taken out of the machine right before, or fed
        # into it right after, mean normal processing, not a manual count.
        touch = _machine_touches(clip, sc, t_start - 0.5, t_end + 0.5)
        checks.append({"check": "machine_touch", "t": t_start, **touch})
        if touch and (touch["after"] >= 3 or touch["before"] >= 10):
            continue
        t = mid["t"]
        score = min(0.95, 0.5 + 0.12 * len(run))
        if score < MANUAL_MIN_CONF:
            continue
        ev.append(Event(
            type="manual_count", t=t, t_start=run[0]["t"] - 0.5, t_end=t_end + 0.5, score=score,
            observation="Кассир пересчитывает банкноты вручную, перелистывая купюры пальцами; счётно-сортировальная "
                        "машина при этом не используется.",
            evidence_t=[t, run[0]["t"], run[-1]["t"]],
            boxes={t: ([(int(obj_boxes[t][0]), int(obj_boxes[t][1]), int(obj_boxes[t][2]), int(obj_boxes[t][3]),
                         "купюры")] if t in obj_boxes else _box_list(mid.get("box")))},
            source="cv+vlm"))
    def detector_sees(labels: tuple, run: list[dict], min_frames: int = 2) -> bool:
        """Corroborate a VLM run with RT-DETR detections inside the cashier crop."""
        n = 0
        for h in run:
            b = h.get("box")
            if not b:
                continue
            for i in range(clip.index_at(h["t"] - 0.6), clip.index_at(h["t"] + 0.6) + 1):
                for d in dets[i]:
                    if d.label in labels and d.score >= 0.4:
                        cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
                        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                            n += 1
        return n >= min_frames

    for run in _runs(hands, lambda h: h.get("phone_in_hands")):
        # Disabled by default: on the training set every "phone" was a desk phone or blue banknotes.
        if not ENABLE_PHONE or len(run) < 3 or not detector_sees(("cell phone",), run):
            continue
        ev.append(Event(
            type="phone_use", t=run[0]["t"], t_start=run[0]["t"] - 0.5, t_end=run[-1]["t"] + 0.5,
            score=min(0.9, 0.5 + 0.15 * len(run)),
            observation="Кассир держит в руках мобильный телефон на рабочем месте.",
            evidence_t=[run[0]["t"], run[-1]["t"]], boxes={run[0]["t"]: _box_list(run[0].get("box"))}, source="vlm"))
    for run in _runs(hands, lambda h: h.get("eating_or_drinking")):
        if len(run) < 2 or not (len(run) >= 4 or detector_sees(("cup", "bottle", "wine glass", "bowl"), run)):
            continue
        ev.append(Event(
            type="food_drink", t=run[0]["t"], t_start=run[0]["t"] - 0.5, t_end=run[-1]["t"] + 0.5,
            score=min(0.9, 0.5 + 0.15 * len(run)),
            observation="Кассир ест или пьёт на рабочем месте, на столе находятся еда/напиток.",
            evidence_t=[run[0]["t"], run[-1]["t"]], boxes={run[0]["t"]: _box_list(run[0].get("box"))}, source="vlm"))
    return ev


# ---------------------------------------------------------------- event checks

def _departure_time(sc: scene_mod.Scene, i_last: int) -> float:
    """Start of the cashier's final walk out: end of her last stationary period before the room emptied."""
    pts = []
    for i in range(max(0, i_last - int(25 * FPS)), i_last + 1):
        b = sc.cashier_box(i)
        if b is not None:
            pts.append((i, (b[0] + b[2]) / 2, b[3], max(1.0, b[3] - b[1])))
    if len(pts) < 6:
        return sc.times[max(0, i_last - int(2 * FPS))]
    p = np.asarray(pts)
    speed = np.zeros(len(p))
    speed[1:] = np.hypot(np.diff(p[:, 1]), np.diff(p[:, 2])) / p[1:, 3] * FPS  # body heights per second
    sm = np.convolve(speed, np.ones(3) / 3, mode="same")
    still = sm < 0.25
    # walk back from the end over the moving tail, then find the end of the last still run (>= 3 frames)
    j = len(p) - 1
    while j >= 0 and not still[j]:
        j -= 1
    k = j
    while k >= 0 and still[k]:
        k -= 1
    if j < 0 or j - k < 3:
        return sc.times[max(0, i_last - int(2 * FPS))]
    return sc.times[int(p[j, 0])]


def _event_checks(clip, sc, dets, hands, debug) -> list[Event]:
    ev: list[Event] = []
    checks: list[dict] = []
    dur = clip.duration
    for e in sc.events:
        try:
            if e["kind"] == "exit" and e["after"] == 0:
                ev += _check_left(clip, sc, e, checks)
            elif e["kind"] == "enter" and e["before"] >= 1:
                ev += _check_entry(clip, sc, e, checks)
            elif e["kind"] == "enter" and e["before"] == 0:
                ev += _check_arrival(clip, sc, e, checks, dur)
            elif ENABLE_CLIENT_LEFT and e["kind"] == "exit" and e["after"] >= 1 and e["before"] >= 2:
                ev += _check_client_left(clip, sc, e, hands, checks)
        except Exception as exc:  # noqa: BLE001
            log.warning("check %s failed: %s", e, exc)
    ev += _check_personal_items(clip, sc, dets, checks)
    if ENABLE_ID_CHECK:
        ev += _check_id_before_cash(clip, sc, hands, checks)
    debug["checks"] = checks
    return ev


def _check_left(clip, sc, e, checks) -> list[Event]:
    out: list[Event] = []
    i_check = min(len(clip.frames) - 1, e["i"] + int(1.0 * FPS))
    t_check = clip.times[i_check]
    t_dep = _departure_time(sc, e["i_prev_end"])
    img = clip.frames[i_check]
    if sc.laptop is not None:
        i_late = min(len(clip.frames) - 1, e["i"] + int(3.0 * FPS))
        votes = []
        for i in sorted({i_check, i_late}):
            r = vlm.ask([_crop(clip.frames[i], sc.laptop, pad=0.4, side=512, enhance=True)], LAPTOP_PROMPT, schema=LAPTOP_SCHEMA,
                        max_side=512, num_predict=30)
            checks.append({"check": "laptop", "t": clip.times[i], "r": r})
            votes.append(isinstance(r, dict) and r.get("screen") == "app")
        if votes and all(votes):
            lb = sc.laptop
            out.append(Event(
                type="pc_unlocked", t=t_dep, t_start=t_dep, t_end=t_check, score=0.7,
                observation="Кассир покинул рабочее место, а экран рабочего компьютера остался разблокирован: "
                            "открыта рабочая программа, а не экран блокировки.",
                evidence_t=[t_dep, t_check],
                boxes={t_check: [(int(lb[0]), int(lb[1]), int(lb[2]), int(lb[3]), "экран")]}, source="cv+vlm"))
    # Ask twice — full frame and desk close-up — and accept either: on the open set the full frame
    # missed a tablet + cups (X-24) and the close-up missed a cash sack next to the machine (X-15).
    r_full = vlm.ask([img], DESK_PROMPT, schema=DESK_SCHEMA, max_side=960, num_predict=80)
    r_crop = vlm.ask([_crop(img, _desk_box(sc), pad=0.05, side=768)], DESK_PROMPT, schema=DESK_SCHEMA,
                     max_side=768, num_predict=80)
    checks.append({"check": "desk_left", "t": t_check, "full": r_full, "crop": r_crop})
    hits = [x for x in (r_full, r_crop) if isinstance(x, dict) and (x.get("items_left") or x.get("banknotes_on_desk"))]
    r = hits[0] if hits else {}
    if isinstance(r, dict) and (r.get("items_left") or r.get("banknotes_on_desk")):
        items = [str(x) for x in (r.get("items") or [])][:4]
        what = "наличные" if r.get("banknotes_on_desk") else ("предметы: " + ", ".join(items) if items else "предметы")
        out.append(Event(
            type="left_unsecured", t=t_dep, t_start=t_dep, t_end=t_check, score=0.65 if r.get("banknotes_on_desk") else 0.55,
            observation=f"Кассир покинул рабочее место; на столе без присмотра остались {what}.",
            evidence_t=[t_dep, t_check], boxes={}, source="cv+vlm"))
    return out


def _check_entry(clip, sc, e, checks) -> list[Event]:
    out: list[Event] = []
    i = min(len(clip.frames) - 1, e["i"] + int(1.0 * FPS))
    img = clip.frames[i]
    r = vlm.ask([_crop(img, _desk_box(sc), pad=0.05, side=768)], CASH_PROMPT, schema=CASH_SCHEMA,
                max_side=768, num_predict=60)
    checks.append({"check": "cash_on_entry", "t": clip.times[i], "r": r})
    if isinstance(r, dict) and r.get("banknotes_lying_on_desk"):
        out.append(Event(
            type="cash_on_desk", t=e["t"], t_start=e["t"], t_end=clip.times[i], score=0.6,
            observation="В кассу входит следующий посетитель, а на столе кассира открыто лежат наличные.",
            evidence_t=[e["t"], clip.times[i]], boxes={}, source="cv+vlm"))
    # "Three people" is not "an outsider": what matters is someone entering the cashier's zone. With a client
    # already served (before >= 2) a moderate approach counts; with the cashier alone the newcomer must come
    # right next to her (a client sits across the desk, an intruder walks behind / beside the cashier).
    reach = 0.9 if e["before"] >= 2 else 0.55
    near = _closest_newcomer(sc, e, reach=reach) if e["hold_end"] - e["t"] >= 3.0 else None
    if near is not None and not _is_real_person(clip, near, checks):
        near = None  # reflection in the glass wall / someone behind the glass, not a person in the room
    if near is not None:
        votes = []
        for dt in (1.0, 2.5):
            j = clip.index_at(e["t"] + dt)
            r = vlm.ask([clip.frames[j]], THIRD_PROMPT, schema=THIRD_SCHEMA, max_side=960, num_predict=20)
            checks.append({"check": "third_person", "t": clip.times[j], "r": r, "before": e["before"]})
            votes.append(isinstance(r, dict) and bool(r.get("person_on_cashier_side")))
        if all(votes):
            out.append(Event(
                type="unauthorized_person", t=e["t"], t_start=e["t"], t_end=e["hold_end"], score=0.55,
                observation=("Во время обслуживания клиента в кассу входит ещё один человек и подходит к кассиру."
                             if e["before"] >= 2 else
                             "В кассу входит человек и проходит в зону кассира, к рабочему месту за столом."),
                evidence_t=[e["t"], clip.times[i]], boxes={},
                source="cv+vlm:third" if e["before"] >= 2 else "cv+vlm:zone"))
    return out


REAL_SCHEMA = {"type": "object", "properties": {"where": {"type": "string", "enum": [
    "inside_room", "reflection_in_glass", "behind_glass_other_room", "unclear"]}}, "required": ["where"]}
REAL_PROMPT = ("Close-up of a person in a small bank cash room with glass walls and a glass door. Is this a real "
               "person physically standing or sitting inside the room (solid, feet on the room floor), a reflection "
               "in the glass (semi-transparent, overlaid on the background behind the glass, mirrored), or a person "
               "behind the glass in another room? Reply in JSON.")


def _closest_newcomer(sc, e, reach: float):
    """(frame index, box) of the non-cashier person who comes closest to the cashier during the new level,
    provided someone stays within `reach` body heights for >= 4 frames; else None."""
    i1 = sc.t_index(e["hold_end"])
    close, best = 0, None
    for i in range(e["i"], i1 + 1):
        cb = sc.cashier_box(i)
        if cb is None:
            continue
        ch = cb[3] - cb[1]
        cc = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
        for d in sc.people[i]:
            if d.box == cb:
                continue
            dc = ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
            dist = np.hypot(dc[0] - cc[0], dc[1] - cc[1]) / max(ch, 1.0)
            if dist < reach:
                close += 1
                if best is None or dist < best[0]:
                    best = (dist, i, d.box)
                break
    return (best[1], best[2]) if close >= 4 and best is not None else None


def _is_real_person(clip, near, checks) -> bool:
    """Ask the VLM, on an enhanced close-up of that exact person, whether they are really in the room."""
    i, box = near
    votes = []
    for di in (0, int(FPS)):
        j = min(len(clip.frames) - 1, i + di)
        crop = _crop(clip.frames[j], box, pad=0.35, side=448, enhance=True)
        try:
            r = vlm.ask([crop], REAL_PROMPT, schema=REAL_SCHEMA, max_side=448, num_predict=20)
        except Exception as exc:  # noqa: BLE001
            log.warning("reflection check failed: %s", exc)
            r = {}
        votes.append(r.get("where") if isinstance(r, dict) else None)
    checks.append({"check": "real_person", "t": clip.times[i], "votes": votes})
    return "inside_room" in votes and not any(v in ("reflection_in_glass", "behind_glass_other_room") for v in votes)


def _someone_close_to_cashier(sc, e, reach: float = 0.9) -> bool:
    """During the new head-count level, does any non-cashier person come within `reach` body heights of her?"""
    i1 = sc.t_index(e["hold_end"])
    close = 0
    for i in range(e["i"], i1 + 1):
        cb = sc.cashier_box(i)
        if cb is None:
            continue
        ch = cb[3] - cb[1]
        cc = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
        for d in sc.people[i]:
            if d.box == cb:
                continue
            dc = ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
            if np.hypot(dc[0] - cc[0], dc[1] - cc[1]) < reach * ch:
                close += 1
                break
    return close >= 4


def _check_arrival(clip, sc, e, checks, dur) -> list[Event]:
    """Cashier walks in: is the door closed behind her a few seconds later?"""
    out: list[Event] = []
    first = sc.people[e["i"]]
    if not first:
        return out
    b = max(first, key=lambda d: d.score).box
    w, h = sc.frame_size
    bw, bh = b[2] - b[0], b[3] - b[1]
    door_box = sc.landmarks.get("door") or (max(0, b[0] - 1.2 * bw), max(0, b[1] - 0.4 * bh),
                                            min(w, b[2] + 1.2 * bw), min(h, b[3] + 0.1 * bh))
    votes = []
    # a careful cashier closes the door 5-8 s after walking in (X-31), so look after that
    for dt in (6.5, 9.0):
        if e["t"] + dt > dur - 0.2:
            continue
        i = clip.index_at(e["t"] + dt)
        r = vlm.ask([_crop(clip.frames[i], door_box, pad=0.05, side=640)], DOOR_PROMPT, schema=DOOR_SCHEMA,
                    max_side=640, num_predict=60)
        checks.append({"check": "door", "t": clip.times[i], "r": r})
        if isinstance(r, dict):
            votes.append(r.get("door") == "open")
    if votes and all(votes):
        out.append(Event(
            type="door_open", t=e["t"] + 1.0, t_start=e["t"], t_end=e["t"] + 6.0, score=0.55,
            observation="Кассир входит в кассу и не закрывает за собой дверь: дверь остаётся открытой.",
            evidence_t=[e["t"] + 1.0, e["t"] + 4.0],
            boxes={e["t"] + 4.0: [(int(door_box[0]), int(door_box[1]), int(door_box[2]), int(door_box[3]), "дверь")]},
            source="cv+vlm"))
    return out


AFTER_SCHEMA = {"type": "object", "properties": {"cashier_handling_cash": {"type": "boolean"}},
                "required": ["cashier_handling_cash"]}
AFTER_PROMPT = ("{n} frames in time order of a bank cashier at her desk; the black box is the banknote counting "
                "machine. Is the cashier counting or handling banknotes in these frames (by hand or by feeding them "
                "through the counting machine)? Reply in JSON.")


def _check_client_left(clip, sc, e, hands, checks) -> list[Event]:
    if e["hold_end"] - e["t"] < 5.0 or e["t"] > clip.duration - 3.0:
        return []  # short dip in the people count (reflections, someone passing), not a real departure
    i0 = clip.index_at(e["t"] + 1.0)
    box = sc.cashier_box(i0)
    if box is None:
        return []
    boxes = [box] + ([sc.landmarks["machine"]] if sc.landmarks.get("machine") is not None else [])
    ub = _union(boxes)
    ts = [e["t"] + dt for dt in (0.5, 1.5, 2.5, 3.5) if e["t"] + dt < clip.duration - 0.2]
    crops = [_crop(clip.frames[clip.index_at(t)], ub, pad=0.1, side=448, pad_y=0.1) for t in ts]
    r = vlm.ask(crops, AFTER_PROMPT.format(n=len(crops)), schema=AFTER_SCHEMA, max_side=448, num_predict=20)
    after = [h for h in hands if e["t"] <= h["t"] <= e["t"] + 5.0 and h.get("present")]
    busy = [h for h in after if h.get("banknotes_in_hands")]
    vlm_busy = isinstance(r, dict) and bool(r.get("cashier_handling_cash"))
    checks.append({"check": "client_left", "t": e["t"], "busy": len(busy), "vlm": r})
    if vlm_busy and len(busy) >= 2:
        return [Event(
            type="count_without_client", t=busy[0]["t"], t_start=e["t"], t_end=busy[-1]["t"], score=0.55,
            observation="Посетитель уходит от кассы, а кассир продолжает операцию с наличными без него.",
            evidence_t=[e["t"], busy[0]["t"]], boxes={busy[0]["t"]: _box_list(busy[0].get("box"))}, source="cv+vlm")]
    return []


ID_CLASSES = ("id_card", "card_or_document")  # an ID card is a card: count both as "document shown"
ID_LOOKBACK_SEC = 30.0                          # ID usually shown shortly before the cash, sometimes at once
ID_LOOKAHEAD_SEC = 3.0


def _client_next_to_cashier(sc, t0: float, t1: float) -> bool:
    """Is a second person (the client) at the desk with the cashier for most of [t0, t1]?"""
    i0, i1 = sc.t_index(t0), sc.t_index(t1)
    near = total = 0
    for i in range(i0, i1 + 1):
        cb = sc.cashier_box(i)
        if cb is None:
            continue
        total += 1
        ch = cb[3] - cb[1]
        cc = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
        for d in sc.people[i]:
            if d.box == cb:
                continue
            dc = ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
            if np.hypot(dc[0] - cc[0], dc[1] - cc[1]) < 1.3 * ch:
                near += 1
                break
    return total > 0 and near / total >= 0.5


def _check_id_before_cash(clip, sc, hands, checks) -> list[Event]:
    """Each cash episode with a client should have an identity document shown around its start.

    Time-based (not visit-based): visit boundaries from head counts are noisy (glass reflections), so for the
    start of every cash episode we look for an ID/card in [start - 30 s, start + 3 s].
    """
    present = [h for h in hands if h.get("present")]
    cash = [h for h in present if h.get("in_hands") == "banknotes"]
    episodes: list[list[dict]] = []
    for h in cash:
        if episodes and h["t"] - episodes[-1][-1]["t"] <= 10.0:
            episodes[-1].append(h)
        else:
            episodes.append([h])
    out: list[Event] = []
    for ep in episodes:
        if len(ep) < 2:
            continue  # a single glimpse of notes is not an operation
        t_b = ep[0]["t"]
        if not _client_next_to_cashier(sc, t_b - 2.0, ep[-1]["t"] + 1.0):
            continue  # cashier alone (processing, packing): no client to identify
        ids = [h for h in present if h.get("in_hands") in ID_CLASSES
               and t_b - ID_LOOKBACK_SEC <= h["t"] <= t_b + ID_LOOKAHEAD_SEC]
        checks.append({"check": "id_before_cash", "first_cash": t_b, "id_seen": [h["t"] for h in ids]})
        if ids:
            continue
        out.append(Event(
            type="no_id_check", t=t_b, t_start=max(0.0, t_b - ID_LOOKBACK_SEC), t_end=t_b, score=0.5,
            observation="Начинается операция с наличными с клиентом; предъявление удостоверения личности "
                        "в предшествующие 30 секунд на видео не зафиксировано.",
            evidence_t=[t_b, max(0.0, t_b - 3.0)], boxes={t_b: _box_list(ep[0].get("box"))},
            source="cv+vlm:id"))
    return out


def _check_personal_items(clip, sc, dets, checks) -> list[Event]:
    """Handbag/backpack detected near the workstation for a while -> ask the VLM to confirm."""
    desk = _desk_box(sc)
    hits = []
    for i, fd in enumerate(dets):
        for d in fd:
            if d.label in ("handbag", "backpack", "suitcase") and d.score >= 0.45:
                cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
                if desk[0] <= cx <= desk[2] and desk[1] <= cy <= desk[3]:
                    hits.append((i, d))
    if len(hits) < int(2 * FPS):
        return []
    i0, d0 = hits[0]
    i = hits[len(hits) // 2][0]
    r = vlm.ask([_crop(clip.frames[i], desk, pad=0.05, side=768)], BAG_PROMPT, schema=BAG_SCHEMA,
                max_side=768, num_predict=60)
    checks.append({"check": "personal_items", "t": clip.times[i], "r": r})
    if isinstance(r, dict) and r.get("personal_items_on_desk"):
        b = d0.box
        return [Event(
            type="personal_items", t=clip.times[i0], t_start=clip.times[i0], t_end=clip.times[hits[-1][0]], score=0.6,
            observation="На рабочем столе кассира лежат личные вещи (сумка/одежда).",
            evidence_t=[clip.times[i0], clip.times[i]],
            boxes={clip.times[i0]: [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), "личные вещи")]}, source="cv+vlm")]
    return []


def _dedupe(events: list[Event]) -> list[Event]:
    """One event per (type, ~5 s window); keep the most confident."""
    events = sorted(events, key=lambda e: -e.score)
    kept: list[Event] = []
    for e in events:
        if any(k.type == e.type and abs(k.t - e.t) < 5.0 for k in kept):
            continue
        kept.append(e)
    return sorted(kept, key=lambda e: e.t)

