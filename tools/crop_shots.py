"""Cut the full-page UI capture into README screenshots (docs/screenshots/)."""
from pathlib import Path

from PIL import Image

D = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
full = D / "_full_result.png"
im = Image.open(full)
print("full", im.size)
im.crop((0, 1150, im.size[0], 3600)).save(D / "result.png", optimize=True)
im.crop((0, 3600, im.size[0], im.size[1])).save(D / "rules-api.png", optimize=True)
full.unlink()
for name in ("_full_home.png", "mobile.png"):
    (D / name).unlink(missing_ok=True)
print(sorted(p.name for p in D.iterdir()))
