# Docker / Singularity Guide

The Docker image provides a fully headless rendering environment using
**EGL** with Mesa's llvmpipe CPU software renderer — no GPU, display server,
or `xvfb` required.  This is fast enough for all snapshot and video tasks.
`libosmesa6` is also included as a last-resort fallback if EGL cannot
initialise.

For **Singularity/Apptainer**, passing `--nv` enables GPU rendering via EGL
automatically — no other configuration needed.

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

## Singularity / Apptainer

The same image can be used with Singularity or Apptainer.

**CPU rendering** (default — no GPU needed):
```bash
singularity exec \
  -B /path/to/subject:/subject \
  -B /path/to/output:/output \
  whippersnappy.sif \
  whippersnap4 \
    -lh /subject/surf/lh.thickness \
    -rh /subject/surf/rh.thickness \
    -sd /subject -o /output/snap4.png
```

**GPU rendering** — pass `--nv` (NVIDIA) or `--rocm` (AMD); EGL selects
the GPU automatically:
```bash
singularity exec --nv \
  -B /path/to/subject:/subject \
  -B /path/to/output:/output \
  whippersnappy.sif \
  whippersnap4 \
    -lh /subject/surf/lh.thickness \
    -rh /subject/surf/rh.thickness \
    -sd /subject -o /output/snap4.png
```

---

## Notes

- The `--init` flag is recommended so that signals (e.g. `Ctrl-C`) are handled
  correctly inside the container.
- `--user $(id -u):$(id -g)` ensures output files are owned by your host user,
  not root.
- The interactive GUI (`whippersnap`) is **not** available in the Docker image —
  it requires a display server and PyQt6, which are not installed.
- **Docker rendering** uses **EGL with CPU software rendering** (Mesa llvmpipe)
  — no GPU or display server required.  The log will show:
  ```
  EGL context active — CPU software rendering (llvmpipe (...), ...)
  ```
- **Singularity GPU rendering** with `--nv` uses EGL with the NVIDIA GPU
  driver injected by Singularity.  The log will show:
  ```
  EGL context active — GPU rendering (...)
  ```
- **OSMesa** (`libosmesa6`) is included as a last-resort CPU fallback for the
  rare case where EGL itself fails to initialise.



