# Docker Guide

The Docker image provides a fully headless rendering environment.  It
automatically uses **EGL** (GPU rendering) when a render device is passed in,
or falls back to **OSMesa** (CPU software renderer) otherwise — no display
server or `xvfb` required in either case.  Both `libegl1` and `libosmesa6`
are pre-installed in the image.

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
- **Default rendering** uses **EGL** (GPU) when `/dev/dri/renderD*` is
  accessible, or **OSMesa** (CPU software renderer, `libosmesa6`) otherwise.
  Both `libegl1` and `libosmesa6` are pre-installed in the image — no extra
  setup is needed.
- **GPU rendering via EGL** requires passing the render device **and** the
  render group into the container.  On the host, `systemd-logind` grants the
  logged-in user direct access to `/dev/dri/renderD*` via a POSIX ACL
  (visible as the `+` in `ls -l`), so no group membership is needed natively.
  Inside Docker there is no login session, so only traditional DAC permissions
  apply — the process must belong to the `render` group to open the device.
  The `--user $(id -u):$(id -g)` flag passes only the primary group; add
  `--group-add` for the render group separately:
  ```bash
  docker run --rm --init \
    --device /dev/dri/renderD128 \
    --group-add render \
    --user $(id -u):$(id -g) \
    -v /path/to/subject:/subject \
    -v /path/to/output:/output \
    whippersnappy \
    -lh /subject/surf/lh.thickness -rh /subject/surf/rh.thickness \
    -sd /subject -o /output/snap4.png
  ```
  The image pre-creates a `render` group with GID 103 (Debian/Ubuntu default).
  If your host uses a different GID, replace `--group-add render` with
  `--group-add $(getent group render | cut -d: -f3)`.
- Without `--device`, WhipperSnapPy falls back to **OSMesa** (CPU) automatically.
  No GPU or `/dev/dri/` device needed for CPU rendering.

