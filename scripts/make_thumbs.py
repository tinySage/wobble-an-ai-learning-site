"""Render one 3D thumbnail per lesson into public/thumbs/. Serve out/ on :8765 first."""
import json, os, re
from playwright.sync_api import sync_playwright
src = open("content/curriculum.ts").read()
slugs = re.findall(r'L\("([a-z0-9-]+)"', src)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 480, "height": 360}, device_scale_factor=1)
    for s in slugs:
        pg.goto(f"http://localhost:8765/studio/?s={s}"); pg.wait_for_selector("#thumb canvas"); pg.wait_for_timeout(900)
        pg.locator("#thumb").screenshot(path=f"public/thumbs/{s}.png", omit_background=True)
    b.close()
print(len(slugs), "thumbnails")

# crop each image to its object, with a little padding, on a fixed 4:3 canvas
from PIL import Image
import glob
for f in glob.glob("public/thumbs/*.png"):
    im = Image.open(f).convert("RGBA"); bb = im.getbbox()
    if not bb: continue
    c = im.crop(bb); w, h = c.size; W = max(w, int(h * 4 / 3)) + 24; H = int(W * 3 / 4)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0)); out.paste(c, ((W - w) // 2, (H - h) // 2), c)
    out.resize((360, 270), Image.LANCZOS).save(f, optimize=True)
