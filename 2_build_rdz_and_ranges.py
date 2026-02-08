import asyncio
import aiohttp
import cv2
import os
import json
import numpy as np
import time

M_TO_PX = 1776 / 1890
RADIUS = 50 * M_TO_PX

BASES = [45, 56, 57, 58]

# CHOOSE ONLINE SERVER WITH ALL REGIONS ACTIVE FOR API DATA, OTHERWISE SCRIPT WILL FAIL
# BASE_URL = "https://war-service-live.foxholeservices.com/api/worldconquest/maps" # ABLE
BASE_URL = "https://war-service-live-2.foxholeservices.com/api/worldconquest/maps" # BAKER/PRE-PATCH
# BASE_URL = "https://war-service-live-3.foxholeservices.com/api/worldconquest/maps" # CHARLIE
# BASE_URL = "https://war-service-dev.foxholeservices.com/api/worldconquest/maps" # DEVBRANCH

REGION_URL = f"{BASE_URL}/{{}}/dynamic/public"

RANGE_ICON_TEMPLATES = {
    28: "templates/obs.png",
    35: "templates/sh.png",
    45: "templates/th.png",
    56: "templates/th.png",
    57: "templates/th.png",
    58: "templates/th.png",
    84: "templates/mh.png",
    53: "templates/cg.png",
}

with open("regions_centre.json", "r") as f:
    CENTRES = json.load(f)

def unnorm(x, y):
    xx = x * 2052.27 - 4.27 / 2
    yy = y * 1776 + 136
    return xx, yy

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

def apply_template_batch(dst, template, coords):
    th, tw = template.shape[:2]
    half_h, half_w = th // 2, tw // 2

    for x_int, y_int in coords:
        x1 = x_int - half_w
        y1 = y_int - half_h
        x2 = x1 + tw
        y2 = y1 + th

        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = tw - max(0, x2 - 2048)
        src_y2 = th - max(0, y2 - 2048)

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(2048, x2)
        dst_y2 = min(2048, y2)

        if src_x2 > src_x1 and src_y2 > src_y1:
            template_crop = template[src_y1:src_y2, src_x1:src_x2]
            roi = dst[dst_y1:dst_y2, dst_x1:dst_x2]
            blend(roi, template_crop)

async def fetch_with_retry(session, url, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    return await response.json()
        except Exception:
            if attempt == retries - 1:
                return None
    return None

async def fetch_regions(session):
    data = await fetch_with_retry(session, BASE_URL)
    return data if data else []

async def fetch_region_data(session, region):
    url = REGION_URL.format(region)
    return region, await fetch_with_retry(session, url)

async def main(ignore_bases=False, ignore_sh=False, ignore_mh=False,
               ignore_obs=False, ignore_cg=False):
    total_start = time.time()

    os.makedirs("output/RDZ", exist_ok=True)
    os.makedirs("output/Ranges", exist_ok=True)

    rdz_template = cv2.imread("rdz_template.png", cv2.IMREAD_UNCHANGED)
    if rdz_template is None:
        raise FileNotFoundError("rdz_template.png not found")
    cv2.imwrite("output/RDZ/HomeRegionW_RDZ.png", rdz_template)
    cv2.imwrite("output/RDZ/HomeRegionC_RDZ.png", rdz_template)
    zr = np.zeros_like(rdz_template)
    cv2.imwrite("output/Ranges/HomeRegionW_ranges.png", zr)
    cv2.imwrite("output/Ranges/HomeRegionC_ranges.png", zr)

    print("\nLoading templates...")
    template_cache = {}
    range_templates = {}
    for icon_type, path in RANGE_ICON_TEMPLATES.items():
        if ignore_bases and icon_type in BASES:
            continue
        if ignore_sh and icon_type == 35:
            continue
        if ignore_mh and icon_type == 84:
            continue
        if ignore_obs and icon_type == 28:
            continue
        if ignore_cg and icon_type == 53:
            continue

        if path not in template_cache:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"{path} not found")
            template_cache[path] = img
        range_templates[icon_type] = template_cache[path]

    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError("mask.png not found")

    async with aiohttp.ClientSession() as session:
        print("\nFetching regions list...")
        regions = await fetch_regions(session)
        total_regions = len(regions)
        print(f"Found {total_regions} regions")

        print("\nFetching region data...")
        tasks = [fetch_region_data(session, region) for region in regions]
        results = await asyncio.gather(*tasks)
        print(f"Fetched data for {total_regions} regions")

        processed_regions = {"homeregionw", "homeregionc"}

        print("\nProcessing regions...")
        t1 = time.time()
        processed_count = 0

        for region, data in results:
            if data is None or "mapItems" not in data:
                continue

            processed_count += 1
            print(f"  {processed_count}/{total_regions}", end="\r")

            processed_regions.add(region.lower())

            rdz_img = rdz_template.copy()
            ranges_img = np.zeros((2048, 2048, 4), dtype=np.uint8)

            coords_by_type = {35: [], 45: [], 56: [], 57: [], 58: [], 28: [], 84: [], 53: []}

            for item in data["mapItems"]:
                icon_type = item.get("iconType")
                x, y = unnorm(item["x"], item["y"])
                x_int, y_int = int(round(x)), int(round(y))

                if icon_type in BASES and not ignore_bases:
                    cv2.circle(rdz_img, (x_int, y_int), int(RADIUS), (0, 0, 0), -1)

                if icon_type in coords_by_type:
                    coords_by_type[icon_type].append((x_int, y_int))

            if not ignore_sh and 35 in range_templates:
                apply_template_batch(ranges_img, range_templates[35],
                                     coords_by_type[35])

            if not ignore_bases:
                for icon_type in [45, 56, 57, 58]:
                    if icon_type not in range_templates:
                        continue
                    apply_template_batch(ranges_img, range_templates[icon_type],
                                         coords_by_type[icon_type])

            if not ignore_obs and 28 in range_templates:
                apply_template_batch(ranges_img, range_templates[28],
                                     coords_by_type[28])

            if not ignore_mh and 84 in range_templates:
                apply_template_batch(ranges_img, range_templates[84],
                                     coords_by_type[84])

            if not ignore_cg and 53 in range_templates:
                cg_img = np.zeros((2048, 2048, 4), dtype=np.uint8)
                apply_template_batch(cg_img, range_templates[53],
                                     coords_by_type[53])
                landscape_path = f"Baked-Mesh-Map/IDLandscape/{region}_id_Landscape.png"
                if os.path.exists(landscape_path):
                    landscape = cv2.imread(landscape_path, cv2.IMREAD_UNCHANGED)
                    if landscape is not None:
                        water_mask = (landscape[:, :, 0] > 100).astype(np.uint8) * 255
                        cg_img[:, :, 3] = cv2.bitwise_and(cg_img[:, :, 3], water_mask)

                blend(ranges_img, cg_img)

            ranges_img[:, :, 3] = cv2.bitwise_and(ranges_img[:, :, 3], mask)

            cv2.imwrite(f"output/RDZ/{region}_RDZ.png", rdz_img)
            cv2.imwrite(f"output/Ranges/{region}_ranges.png", ranges_img)

        print(f"\nProcessed {processed_count} regions in {time.time() - t1:.2f}s")

        missing_regions = set(CENTRES.keys()) - processed_regions
        if missing_regions:
            print(f"\nMissing regions ({len(missing_regions)}):")
            for region in sorted(missing_regions):
                print(f"  {region}")

    print(f"\nDone in {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main(
        ignore_bases=False,
        ignore_sh=False,
        ignore_mh=False,
        ignore_obs=False,
        ignore_cg=False
    ))
