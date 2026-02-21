# WhipperSnapPy

WhipperSnapPy is a Python/OpenGL tool to render triangular surface meshes
with color overlays or parcellations and generate screenshots — from the
command line, in Jupyter notebooks, or via a desktop GUI.

It works with FreeSurfer and FastSurfer brain surfaces as well as any
triangle mesh in OFF, legacy ASCII VTK PolyData, or ASCII PLY format, or
passed directly as a NumPy ``(vertices, faces)`` tuple.

## Installation

```bash
pip install whippersnappy
```

For rotation video support (MP4/WebM):

```bash
pip install 'whippersnappy[video]'
```

For the interactive desktop GUI:

```bash
pip install 'whippersnappy[gui]'
```

For interactive 3D in Jupyter notebooks:

```bash
pip install 'whippersnappy[notebook]'
```

Off-screen (headless) rendering is supported natively via EGL on Linux — no
`xvfb` required. See the <a href="DOCKER.md">Docker guide</a> for headless usage.

## Command-Line Usage

After installation the following commands are available:

### Four-view snapshot (`whippersnap4`)

Renders lateral and medial views of both hemispheres into a single composed image:

```bash
whippersnap4 -lh $LH_OVERLAY \
             -rh $RH_OVERLAY \
             -sd $SUBJECT_DIR \
             --fmax 4 --fthresh 2 \
             --caption "Cortical Thickness" \
             -o snap4.png
```

### Single-view snapshot (`whippersnap1`)

Renders one view of any triangular surface mesh:

```bash
whippersnap1 $SUBJECT_DIR/surf/lh.white \
             --overlay $LH_OVERLAY \
             --bg-map  $SUBJECT_DIR/surf/lh.curv \
             --roi     $SUBJECT_DIR/label/lh.cortex.label \
             --view left \
             -o snap1.png

# Also works with OFF / VTK / PLY
whippersnap1 mesh.off --overlay values.mgh -o snap1.png
```

### Rotation video (`whippersnap1 --rotate`)

Renders a 360° animation of any triangular surface mesh:

```bash
whippersnap1 $SUBJECT_DIR/surf/lh.white \
             --overlay $LH_OVERLAY \
             --rotate \
             -o rotation.mp4
```

### Desktop GUI (`whippersnap`)

Launches an interactive Qt window with live threshold controls.

**General mode** — any triangular mesh:

```bash
pip install 'whippersnappy[gui]'
whippersnap --mesh mesh.off --overlay values.mgh
whippersnap --mesh lh.white --overlay lh.thickness --bg-map lh.curv
```

**FreeSurfer shortcut** — derive all paths from a subject directory:

```bash
whippersnap -sd $SUBJECT_DIR --hemi lh -lh $LH_OVERLAY
whippersnap -sd $SUBJECT_DIR --hemi rh --annot rh.aparc.annot
```

For all options run `whippersnap4 --help`, `whippersnap1 --help`, or `whippersnap --help`.

## Python API

```python
from whippersnappy import snap1, snap4, snap_rotate, plot3d
```

| Function | Description |
|---|---|
| `snap1` | Single-view snapshot of any triangular mesh → PIL Image |
| `snap4` | Four-view composed image (FreeSurfer subject, lateral/medial both hemispheres) |
| `snap_rotate` | 360° rotation video of any triangular mesh (MP4, WebM, or GIF) |
| `plot3d` | Interactive 3D WebGL viewer for Jupyter notebooks |

**Supported mesh inputs for `snap1`, `snap_rotate`, and `plot3d`:**
FreeSurfer binary surfaces (e.g. `lh.white`), OFF (`.off`), legacy ASCII VTK PolyData (`.vtk`), ASCII PLY (`.ply`), or a `(vertices, faces)` NumPy array tuple.

### Example

```python
from whippersnappy import snap1, snap4

# FreeSurfer surface with overlay
img = snap1('lh.white',
            overlay='lh.thickness',
            bg_map='lh.curv',
            roi='lh.cortex.label')
img.save('snap1.png')

# Four-view overview (FreeSurfer subject directory)
img = snap4(sdir='/path/to/subject',
            lh_overlay='/path/to/lh.thickness',
            rh_overlay='/path/to/rh.thickness',
            colorbar=True, caption='Cortical Thickness (mm)')
img.save('snap4.png')

# OFF / VTK / PLY mesh
img = snap1('mesh.off', overlay='values.mgh')

# Array inputs (e.g. from LaPy or trimesh)
import numpy as np
v = np.random.randn(1000, 3).astype(np.float32)
f = np.array([[0, 1, 2]], dtype=np.uint32)
overlay = np.random.randn(1000).astype(np.float32)
img = snap1((v, f), overlay=overlay)
```

CLI usage:

```bash
# Single view
whippersnap1 lh.white --overlay lh.thickness --bg-map lh.curv --roi lh.cortex.label -o snap1.png

# Four-view batch
whippersnap4 -lh lh.thickness -rh rh.thickness -sd /path/to/subject -o snap4.png
```

See `tutorials/whippersnappy_tutorial.ipynb` for complete notebook examples.


## Docker

The Docker image provides a fully headless EGL rendering environment — no
display server or `xvfb` required. See <a href="DOCKER.md"><strong>DOCKER.md</strong></a> for details.

## API Documentation

https://deep-mi.org/WhipperSnapPy

## Links

Lab webpage: https://deep-mi.org
