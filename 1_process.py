import cv2
import json
import numpy as np
import os
import time

# 1 step = 0.5 meters
# 5 steps = 2.5 meters
# 10 steps = 5 meters
CONTOURS_STEP = 5

with open("regions_centre.json", "r") as f:
    CENTRES_REF = json.load(f)
with open("regions_water.json", "r") as f:
    WATER_REF = json.load(f)
with open("regions_eq_level.json", "r") as f:
    EQ_LEVEL_REF = json.load(f)
MASK = cv2.threshold(cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE), 127, 1, cv2.THRESH_BINARY)[1]
MASK3 = MASK[:, :, np.newaxis]

def generate_palette(fns):
    colors = {}
    n = len(fns)
    for i, fn in enumerate(fns):
        hue = int((i * 180 / n) % 180)
        hsv = np.array([[[hue, 63, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors[fn] = tuple(map(int, bgr))
    return colors

COLORS = generate_palette(CENTRES_REF)
COLORS_INV = {v: k for k, v in COLORS.items()}

def norm(img):
    vals = []
    for i in range(10):
        for j in range(10):
            vals.append([i * 10 + j + 1, float(img[10 + i * 21, 10 + j * 21])])
    while abs(vals[0][1] - vals[1][1]) < 0.001:
        vals.pop(0)
    while abs(vals[-1][1] - vals[-2][1]) < 0.001:
        vals.pop(-1)
    vals.pop(-1)
    vals.pop(0)
    mn, mnd = vals[0]
    mx, mxd = vals[-1]
    img *= (mx - mn) / (mxd - mnd)
    img[:250, :250] = -30

def load_patches():
    start = time.time()
    print("\nLoading patches...")
    total = len(CENTRES_REF)

    patches = {}
    for i, (fn, coords) in enumerate(CENTRES_REF.items(), 1):
        img = cv2.imread(f"Baked-Mesh-Map/Height/{fn}_height.tif",
                         cv2.IMREAD_UNCHANGED).astype(np.float32)
        mask_low = img <= 0.5
        norm(img)
        if fn in WATER_REF:
            ox, oy = WATER_REF[fn]
            img -= img[oy, ox]
        img[mask_low] = -30
        patches[fn] = (*coords, img)
        print(f"  {i}/{total}", end="\r")

    for fn, (_, _, img) in list(patches.items()):
        if fn in EQ_LEVEL_REF:
            c, fn1, c1 = EQ_LEVEL_REF[fn]
            aa1, aa2 = img[*c], patches[fn1][2][*c1]
            img -= aa1 - aa2
        img[np.abs(img) < 0.001] = 0
        landscape = cv2.imread(f"Baked-Mesh-Map/IDLandscape/{fn}_id_Landscape.png",
                               cv2.IMREAD_UNCHANGED)
        # img[MASK < 1] = -30
        img[((landscape[:, :, 2] > 100) | MASK) < 1] = -30

    print(f"\nPatches loaded in {time.time() - start:.2f}s")
    return [(fn, x, y, img, COLORS[fn]) for fn, (x, y, img) in patches.items()]

def heights_and_sources():
    os.makedirs("output", exist_ok=True)
    with open("output/sources.json", "w", encoding="utf-8") as f:
        json.dump({k: list(reversed(v)) for k, v in COLORS.items()},
                  f, indent=4, ensure_ascii=False)

    patches = load_patches()
    sz = [0, 0]
    for _, x, y, _, _ in patches:
        sz[0] = max(sz[0], y + 1024)
        sz[1] = max(sz[1], x + 1024)

    canvas = np.full(sz, -30.0, dtype=np.float32)
    extents = np.zeros(sz, dtype=np.uint8)
    sourcemap = np.full((*sz, 3), [0, 0, 0], dtype=np.uint8)
    for fn, x, y, img, col in patches:
        region        =    canvas[y - 1024:y + 1024, x - 1024:x + 1024]
        source_region = sourcemap[y - 1024:y + 1024, x - 1024:x + 1024]
        update_mask = img > region
        region[update_mask] = img[update_mask]
        source_region[update_mask] = col
        extents[y - 1024:y + 1024, x - 1024:x + 1024] |= MASK

    canvas[extents == 0] = -30
    sourcemap[extents == 0] = [0, 0, 0]
    heightmap = 60 + canvas * 2
    heightmap = np.clip(heightmap, 0, 255).astype(np.uint8)

    t1 = time.time()
    cv2.imwrite("output/heightmap.png", heightmap)
    cv2.imwrite("output/sourcemap.png", sourcemap)

    t1 = time.time()
    os.makedirs("output/Spills", exist_ok=True)
    region_masks = {}
    total = len(patches)
    print(f"\nProcessing Spills...")
    for i, (fn, x, y, _, _) in enumerate(patches, 1):
        region_mask = (sourcemap[y - 1024:y + 1024, x - 1024:x + 1024] == COLORS[fn]).all(axis=2)
        region_masks[fn] = region_mask
        cv2.imwrite(f"output/Spills/{fn}.png", region_mask.astype(np.uint8) * 255)
        print(f"  {i}/{total}", end="\r")
    print(f"\nWritten Spills in {time.time() - t1:.2f}s")

    t1 = time.time()
    os.makedirs("output/Height", exist_ok=True)
    print(f"\nProcessing Height...")
    for i, (fn, x, y, _, _) in enumerate(patches, 1):
        region_heightmap = heightmap[y - 1024:y + 1024, x - 1024:x + 1024] * MASK
        cv2.imwrite(f"output/Height/{fn}.png", region_heightmap)
        print(f"  {i}/{total}", end="\r")
    print(f"\nWritten Height in {time.time() - t1:.2f}s")
    return sz, region_masks, heightmap

def extras(sz, region_masks):
    folders = [
        ("IDLandscape", "landscape", "_id_Landscape.png"),
        ("IDRoads", "roads", "_id_Roads.png"),
        ("NormalMap_OpenGL", "norm", "_normal_opengl.png"),
        ("Curvature", "curvature", "_curvature.png"),
        ("AmbientOcclusion", "ao", "_ao.png"),
    ]

    patches = [(fn, *coords) for fn, coords in CENTRES_REF.items()]
    landscape_canvas = None

    for folder, ofn, suffix in folders:
        t1 = time.time()
        print(f"\nProcessing {folder}...")
        canvas = None

        for fn, x, y in patches:
            imgn = f"Baked-Mesh-Map/{folder}/{fn}{suffix}"
            img = cv2.imread(imgn, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Missing {imgn}")
                continue
            if canvas is None:
                if img.ndim == 3:
                    canvas = np.zeros((*sz, img.shape[2]), dtype=img.dtype)
                else:
                    canvas = np.zeros(sz, dtype=img.dtype)

            region = canvas[y - 1024:y + 1024, x - 1024:x + 1024]
            mask = region_masks[fn]
            region[mask] = img[mask]

        if canvas is None:
            print(f"No images found for {folder}, skipping")
            continue

        if ofn in ["ao", "roads"] and landscape_canvas is not None:
            mask_transparent = landscape_canvas[:, :, 0] > 100
            canvas[mask_transparent] = 0
        elif ofn == "landscape":
            landscape_canvas = canvas

        cv2.imwrite(f"output/{ofn}.png", canvas)
        os.makedirs(f"output/{folder}", exist_ok=True)
        for i, (fn, x, y) in enumerate(patches, 1):
            extracted = canvas[y - 1024:y + 1024, x - 1024:x + 1024]
            if extracted.ndim == 3:
                extracted = extracted * MASK3
            else:
                extracted = extracted * MASK
            cv2.imwrite(f"output/{folder}/{fn}{suffix}", extracted)
            print(f"  {i}/{len(patches)}", end="\r")
        print(f"\nWritten {folder} in {time.time() - t1:.2f}s")

    return landscape_canvas

def contours(heightmap, landscape_canvas):
    start = time.time()
    print(f"\nProcessing Contours...")
    total = len(CENTRES_REF)

    levels = (heightmap - 1) // CONTOURS_STEP
    padded_levels = np.pad(levels, 1, mode="edge")
    contour_mask = np.zeros_like(heightmap, dtype=bool)

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = padded_levels[1+dy:1+dy+levels.shape[0], 1+dx:1+dx+levels.shape[1]]
        contour_mask |= levels > shifted

    contour_img = (contour_mask * 255).astype(np.uint8)

    if landscape_canvas is not None:
        contour_img[landscape_canvas[:, :, 1] < 100] = 0

    cv2.imwrite("output/contours.png", contour_img)
    os.makedirs("output/Contours", exist_ok=True)
    for i, (fn, coords) in enumerate(CENTRES_REF.items(), 1):
        x, y = coords
        region_contours = contour_img[y - 1024:y + 1024, x - 1024:x + 1024] * MASK
        cv2.imwrite(f"output/Contours/{fn}_contours.png", region_contours)
        print(f"  {i}/{total}", end="\r")

    print(f"\nWritten Contours in {time.time() - start:.2f}s")

if __name__ == "__main__":
    total_start = time.time()
    sz, region_masks, heightmap = heights_and_sources()
    landscape_canvas = extras(sz, region_masks)
    contours(heightmap, landscape_canvas)
    print(f"Done in {time.time() - total_start:.2f}s")
