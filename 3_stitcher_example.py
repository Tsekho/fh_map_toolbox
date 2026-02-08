import cv2
import json
import numpy as np
import os
import time

with open("regions_centre.json", "r") as f:
    CENTRES = json.load(f)

def calculate_canvas_size():
    max_y, max_x = 0, 0
    for x, y in CENTRES.values():
        max_y = max(max_y, y + 1024)
        max_x = max(max_x, x + 1024)
    return max_y, max_x

def blend(src, img):
    if img.ndim == 2:
        src[:, :, 3] = np.maximum(src[:, :, 3], img)
        return

    src_alpha = src[:, :, 3] / 255.0
    img_alpha = img[:, :, 3] / 255.0

    out_alpha = src_alpha + img_alpha * (1 - src_alpha)

    mask = out_alpha > 0

    src_rgb = src[:, :, :3].astype(np.float32)
    img_rgb = img[:, :, :3].astype(np.float32)

    out_rgb = np.zeros_like(src_rgb)
    out_rgb[mask] = (
        src_rgb[mask] * src_alpha[mask, np.newaxis]
        + img_rgb[mask] * img_alpha[mask, np.newaxis] * (1 - src_alpha[mask, np.newaxis])
    ) / out_alpha[mask, np.newaxis]

    src[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    src[:, :, 3] = np.clip(out_alpha * 255, 0, 255).astype(np.uint8)

def stitch(pattern, output):
    t1 = time.time()
    print(f"\nStitching {output}...")
    
    height, width = calculate_canvas_size()
    canvas_rgb = np.zeros((height, width, 3), dtype=np.float32)
    canvas_alpha = np.zeros((height, width), dtype=np.float32)
    alpha_sum = np.zeros((height, width), dtype=np.float32)

    total = len(CENTRES)
    processed = 0

    for fn, (x, y) in CENTRES.items():
        img_path = pattern.format(fn)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"\nMissing: {img_path}")
            continue

        processed += 1
        print(f"  {processed}/{total}", end="\r")

        img = img.astype(np.float32)
        alpha = img[:, :, 3] / 255.0
        rgb = img[:, :, :3]

        y1, y2 = y - 1024, y + 1024
        x1, x2 = x - 1024, x + 1024

        canvas_rgb[y1:y2, x1:x2] += rgb * alpha[:, :, np.newaxis]
        canvas_alpha[y1:y2, x1:x2] = np.maximum(canvas_alpha[y1:y2, x1:x2], img[:, :, 3])
        alpha_sum[y1:y2, x1:x2] += alpha

    mask = alpha_sum > 0
    canvas_rgb[mask] /= alpha_sum[mask, np.newaxis]

    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    canvas[:, :, :3] = np.clip(canvas_rgb, 0, 255).astype(np.uint8)
    canvas[:, :, 3] = np.clip(canvas_alpha, 0, 255).astype(np.uint8)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite(f"output/{output}", canvas)
    
    print(f"\nWritten {output} in {time.time() - t1:.2f}s")

if __name__ == "__main__":
    total_start = time.time()
    stitch("output/RDZ/{}_RDZ.png", "rdz.png")
    stitch("border.png", "hex_grid.png")
    stitch("output/Ranges/{}_ranges.png", "ranges.png")
    print(f"\nDone in {time.time() - total_start:.2f}s")
