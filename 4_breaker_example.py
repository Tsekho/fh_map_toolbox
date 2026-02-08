import cv2
import json
import numpy as np
import os
import time

with open("regions_centre.json", "r") as f:
    CENTRES = json.load(f)

def break_image(output_pattern, stitched_path):
    t1 = time.time()
    print(f"\nBreaking {stitched_path}...")
    
    stitched = cv2.imread(stitched_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

    if stitched is None:
        print(f"Error: Could not load {stitched_path}")
        return

    if mask is None:
        print(f"Error: Could not load mask.png")
        return

    total = len(CENTRES)
    
    for i, (region_name, (x, y)) in enumerate(CENTRES.items(), 1):
        y1, y2 = y - 1024, y + 1024
        x1, x2 = x - 1024, x + 1024

        region_img = stitched[y1:y2, x1:x2].copy()

        if region_img.shape[2] == 4:
            region_img[:, :, 3] = np.minimum(region_img[:, :, 3], mask)
        else:
            region_img = np.dstack([region_img, mask])

        output_path = output_pattern.format(region_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, region_img)
        
        print(f"  {i}/{total}", end="\r")
    
    print(f"\nBroken {stitched_path} in {time.time() - t1:.2f}s")

if __name__ == "__main__":
    total_start = time.time()
    break_image("output/Sources/{}_source.png", "output/sourcemap.png")
    print(f"\nDone in {time.time() - total_start:.2f}s")
