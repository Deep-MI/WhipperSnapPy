# Docker Guide

The Docker image provides a fully headless rendering environment using
**EGL** — no display server or `xvfb` required.  Without `--device`, EGL
renders via Mesa's llvmpipe (CPU); with `--device /dev/dri/renderD128` it
uses the GPU.  `libosmesa6` is also included as a last-resort fallback for
the rare case where EGL itself cannot initialise.

The default entry point is `whippersnap4` (four-view batch rendering).
`whippersnap1` (single-view snapshot and rotation video) can be invoked by
overriding the entry point.

---

## Building the image

From the repository root:

```bash
docker build --rm -t whippersnappy -f Dockerfile .
```

---

## Running — four-view batch rendering (`whippersnap4`)

`whippersnap4` renders lateral and medial views of both hemispheres and writes
a single composed PNG image.

Mount your local directories into the container and pass the in-container paths
as arguments:

```bash
docker run --rm --init \
  -v /path/to/subject:/subject \
  -v /path/to/output:/output \
  --user $(id -u):$(id -g) \
  whippersnappy \
  -lh /subject/surf/lh.thickness \
  -rh /subject/surf/rh.thickness \
  -sd /subject \
  -o /output/snap4.png
```

### With an annotation file instead of an overlay

```bash
docker run --rm --init \
  -v /path/to/subject:/subject \
  -v /path/to/output:/output \
  --user $(id -u):$(id -g) \
  whippersnappy \
  --lh_annot /subject/label/lh.aparc.annot \
  --rh_annot /subject/label/rh.aparc.annot \
  -sd /subject \
  -o /output/snap4_annot.png
```

### With a caption and custom thresholds

```bash
docker run --rm --init \
  -v /path/to/subject:/subject \
  -v /path/to/output:/output \
  --user $(id -u):$(id -g) \
  whippersnappy \
  -lh /subject/surf/lh.thickness \
  -rh /subject/surf/rh.thickness \
  -sd /subject \
  --fthresh 2.0 --fmax 4.0 \
  --caption "Cortical thickness" \
  -o /output/snap4_thickness.png
```

### All `whippersnap4` options

```
docker run --rm whippersnappy --help
```

---

## Running — single-view snapshot (`whippersnap1`)

Override the entry point with `--entrypoint whippersnap1`.  Any triangular
mesh format is supported: FreeSurfer binary surfaces, OFF, ASCII VTK
PolyData, ASCII PLY, and GIfTI (`.gii`, `.surf.gii`).

### FreeSurfer surface

```bash
docker run --rm --init \
  --entrypoint whippersnap1 \
  -v /path/to/subject:/subject \
  -v /path/to/output:/output \
  --user $(id -u):$(id -g) \
  whippersnappy \
  --mesh /subject/surf/lh.white \
  --overlay /subject/surf/lh.thickness \
  --bg-map /subject/surf/lh.curv \
  --roi /subject/label/lh.cortex.label \
  --view left \
  --fthresh 2.0 --fmax 4.0 \
  -o /output/snap1.png
```

### OFF / VTK / PLY / GIfTI mesh

```bash
docker run --rm --init \
  --entrypoint whippersnap1 \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  --user $(id -u):$(id -g) \
  whippersnappy \
  --mesh /data/mesh.off \
  --overlay /data/values.txt \
  -o /output/snap1.png
```

### All `whippersnap1` options

```bash
docker run --rm --entrypoint whippersnap1 whippersnappy --help
```

---

## Running — 360° rotation video (`whippersnap1 --rotate`)

`whippersnap1 --rotate` renders a full 360° rotation video and writes an
`.mp4`, `.webm`, or `.gif` file.  `imageio-ffmpeg` is bundled in the image —
no system ffmpeg is required.

### MP4 (H.264, recommended)

```bash
docker run --rm --init \
  --entrypoint whippersnap1 \
  -v /path/to/subject:/subject \
  -v /path/to/output:/output \
  --user $(id -u):$(id -g) \
  whippersnappy \
  --mesh /subject/surf/lh.white \
  --overlay /subject/surf/lh.thickness \
  --bg-map /subject/surf/lh.curv \
  --rotate \
  --rotate-frames 72 \
  --rotate-fps 24 \
  -o /output/rotation.mp4
```

### Animated GIF (no ffmpeg needed)

```bash
docker run --rm --init \
  --entrypoint whippersnap1 \
  -v /path/to/subject:/subject \
  -v /path/to/output:/output \
  --user $(id -u):$(id -g) \
  whippersnappy \
  --mesh /subject/surf/lh.white \
  --overlay /subject/surf/lh.thickness \
  --rotate \
  --rotate-frames 36 \
  --rotate-fps 12 \
  -o /output/rotation.gif
```

---

## Path mapping summary

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `/path/to/subject` | `/subject` | FreeSurfer subject directory (contains `surf/`, `label/`) |
| `/path/to/output` | `/output` | Directory where output files are written |

All output files are written to the container path you pass via `-o`; mount the
parent directory to retrieve them on the host.

---

## Notes

- The `--init` flag is recommended so that signals (e.g. `Ctrl-C`) are handled
  correctly inside the container.
- `--user $(id -u):$(id -g)` ensures output files are owned by your host user,
  not root.
- The interactive GUI (`whippersnap`) is **not** available in the Docker image —
  it requires a display server and PyQt6, which are not installed.
- **Default rendering** uses **EGL**.  Without `--device`, EGL uses Mesa's
  software rasterizer (llvmpipe) for CPU rendering — no GPU required.  With
  `--device /dev/dri/renderD128`, EGL uses the GPU automatically.  `libegl1`
  is pre-installed in the image; no extra flags required.
- **OSMesa** is included (`libosmesa6`) as a last-resort fallback only for
  the rare case where EGL itself fails to initialise (e.g. a stripped-down
  base image without Mesa's EGL backend).  Under normal circumstances EGL
  handles both the GPU and CPU rendering paths.
- **GPU rendering** is selected automatically by EGL when you pass the render
  device into the container (optional — only needed for hardware acceleration):
  ```bash
  docker run --rm --init \
    --device /dev/dri/renderD128 \
    --user $(id -u):$(id -g) \
    -v /path/to/subject:/subject \
    -v /path/to/output:/output \
    whippersnappy \
    -lh /subject/surf/lh.thickness -rh /subject/surf/rh.thickness \
    -sd /subject -o /output/snap4.png
  ```


