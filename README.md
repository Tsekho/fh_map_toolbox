# Foxhole Map Processing Tools

Collection of Python scripts for processing, stitching, and enhancing map layers from [Wolfgang-IX/Foxhole-Map-Project](https://github.com/Wolfgang-IX/Foxhole-Map-Project)

## Requirements

### Python Dependencies

```bash
pip install aiohttp opencv-python numpy
```

## Scripts

### 1. `0_download_wolfgang.py`

Downloads `Wolfgang-IX/Foxhole-Map-Project/Images/Baked-Mesh-Map` repository data.

### 2. `1_process.py`

Processes raw map data with the following operations:

- **Normalizes heightmaps**
  - 60 is the global water level
  - 1 shade of gray = 0.5m
- **Builds contour lines** (per 2.5m or custom elevation step)
- **Applies rock spilling fix** to the following layers:
  - IDLandscape
  - IDRoads
  - NormalMap_OpenGL
  - Curvature
  - AmbientOcclusion

> **Rock Spilling Fix:** Many rocks and mountains near borders belong to one region and not the other, which makes region renders inaccurate and results in visible stitches. This fix greatly negates this effect by allowing rock renders to spill over into neighboring regions.

### 3. `2_build_rdz_and_ranges.py`

Polls API and builds RDZ and structures ranges layers (bases, SH, MG, CG and OBS).

- Supports custom templates
- Allows choosing which structures to ignore
- Existing templates are accurately scaled and can be used as reference

### 4. `3_stitcher_example.py`

Example script that stitches together per-region images.

### 5. `4_breaker_example.py`

Example script that takes stitched image and breaks it apart into per-region images.
