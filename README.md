# WhipperSnapPy

WhipperSnapPy is a Python/OpenGL tool to render triangular surface meshes
with color overlays or parcellations and generate screenshots — from the
command line, in Jupyter notebooks, or via a desktop GUI.

It works with FreeSurfer and FastSurfer brain surfaces as well as any
triangle mesh in OFF, legacy ASCII VTK PolyData, ASCII PLY, or GIfTI
(`.gii`, `.surf.gii`) format, or passed directly as a NumPy
`(vertices, faces)` tuple.

## Installation

```bash
pip install whippersnappy
```

For rotation video support (MP4/WebM — GIF works without this):

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

Off-screen (headless) rendering on **Linux** is supported natively via
OSMesa — no `xvfb` or GPU required.  On **macOS** and **Windows** a real
display connection is needed (GLFW creates an invisible window backed by the
system GPU driver).  See the <a href="DOCKER.md">Docker guide</a> for
headless Linux usage.

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
whippersnap1 --mesh $SUBJECT_DIR/surf/lh.white \
             --overlay $LH_OVERLAY \
             --bg-map  $SUBJECT_DIR/surf/lh.curv \
             --roi     $SUBJECT_DIR/label/lh.cortex.label \
             --view left \
             -o snap1.png

# Also works with OFF / VTK / PLY / GIfTI
whippersnap1 --mesh mesh.off --overlay values.txt -o snap1.png
whippersnap1 --mesh surface.surf.gii --overlay overlay.func.gii -o snap1.png
```

### Rotation video (`whippersnap1 --rotate`)

Renders a 360° animation of any triangular surface mesh.  GIF output uses
pure PIL (no extra install); MP4/WebM requires `pip install 'whippersnappy[video]'`.

```bash
whippersnap1 --mesh $SUBJECT_DIR/surf/lh.white \
             --overlay $LH_OVERLAY \
             --rotate \
             -o rotation.mp4

whippersnap1 --mesh $SUBJECT_DIR/surf/lh.white \
             --rotate \
             -o rotation.gif
```

### Desktop GUI (`whippersnap`)

Launches an interactive Qt window with live threshold controls and
mouse-driven rotation, pan, and zoom.  Requires
`pip install 'whippersnappy[gui]'`.

**General mode** — any triangular mesh:

```bash
whippersnap --mesh mesh.off --overlay values.txt
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
from whippersnappy import snap1, snap4, snap_rotate, ViewType
from whippersnappy import plot3d  # requires whippersnappy[notebook]
```

| Function / Class | Description |
|---|---|
| `snap1` | Single-view snapshot of any triangular mesh → PIL Image |
| `snap4` | Four-view composed image (FreeSurfer subject, lateral/medial both hemispheres) |
| `snap_rotate` | 360° rotation video of any triangular surface mesh (MP4, WebM, or GIF) |
| `plot3d` | Interactive 3D WebGL viewer for Jupyter notebooks |
| `ViewType` | Enum of camera presets used by `snap1` and `snap_rotate` |

**`ViewType` values** — pass to the `view` parameter of `snap1` or the
`start_view` parameter of `snap_rotate`:

| Value | Description |
|---|---|
| `ViewType.LEFT` | Left lateral view *(default)* |
| `ViewType.RIGHT` | Right lateral view |
| `ViewType.FRONT` | Frontal / anterior view |
| `ViewType.BACK` | Posterior view |
| `ViewType.TOP` | Superior / dorsal view |
| `ViewType.BOTTOM` | Inferior / ventral view |

**Supported mesh inputs for `snap1`, `snap_rotate`, and `plot3d`:**
FreeSurfer binary surfaces (e.g. `lh.white`), OFF (`.off`), legacy ASCII
VTK PolyData (`.vtk`), ASCII PLY (`.ply`), GIfTI surface
(`.gii`, `.surf.gii`), or a `(vertices, faces)` NumPy array tuple.

**Supported overlay/label inputs:**
FreeSurfer morph (`.curv`, `.thickness`), MGH/MGZ (`.mgh`, `.mgz`),
plain text (`.txt`, `.csv`), NumPy (`.npy`, `.npz`),
GIfTI functional/label (`.func.gii`, `.label.gii`).

### Examples

```python
from whippersnappy import snap1, snap4, ViewType

# FreeSurfer surface with overlay — default left lateral view
img = snap1('lh.white',
            overlay='lh.thickness',
            bg_map='lh.curv',
            roi='lh.cortex.label')
img.save('snap1.png')

# Specific view
img = snap1('lh.white', overlay='lh.thickness', view=ViewType.FRONT)
img.save('snap1_front.png')

# Four-view overview (FreeSurfer subject directory)
img = snap4(lh_overlay='/path/to/lh.thickness',
            rh_overlay='/path/to/rh.thickness',
            sdir='/path/to/subject',
            colorbar=True,
            caption='Cortical Thickness (mm)')
img.save('snap4.png')

# OFF / VTK / PLY / GIfTI mesh
img = snap1('mesh.off', overlay='values.txt')
img = snap1('surface.surf.gii', overlay='overlay.func.gii')

# Array inputs (e.g. from LaPy or trimesh)
import numpy as np
v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
f = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], dtype=np.uint32)
overlay = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
img = snap1((v, f), overlay=overlay)
```

See `tutorials/whippersnappy_tutorial.ipynb` for complete notebook examples.


## Docker

The Docker image provides a fully headless OSMesa rendering environment — no
display server, `xvfb`, or GPU required. See <a href="DOCKER.md"><strong>DOCKER.md</strong></a> for details.

## API Documentation

https://deep-mi.org/WhipperSnapPy

## Links

Lab webpage: https://deep-mi.org
