# WhipperSnapPy

WhipperSnapPy is a Python OpenGL program to render FreeSurfer and
FastSurfer surface models with color overlays or parcellations and generate
screenshots.

## Contents:

- `snap1` — single-view surface snapshot
- `snap4` — four-view composed image (lateral/medial, both hemispheres)
- `snap_rotate` — 360° rotation video (MP4, WebM, or GIF)
- `plot3d` — interactive 3D WebGL viewer for Jupyter notebooks
- `whippersnap` — desktop GUI with live Qt controls

## Installation:

The `WhipperSnapPy` package can be installed from PyPI via:

```bash
python3 -m pip install whippersnappy
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

## Usage:

### Local:

After installing the Python package, the command-line tools can be run as in
the following examples:

```bash
# Four-view batch rendering (both hemispheres)
whippersnap4 -lh $LH_OVERLAY -rh $RH_OVERLAY \
             -sd $SURF_SUBJECT_DIR \
             --fmax 4 --fthresh 2 --invert \
             --caption "My caption" \
             -o $OUTPUT_DIR/snap4.png

# Single-view snapshot
whippersnap1 $SURF_SUBJECT_DIR/surf/lh.white \
             --overlay $LH_OVERLAY \
             --view left -o $OUTPUT_DIR/snap1.png

# 360° rotation video
whippersnap1 $SURF_SUBJECT_DIR/surf/lh.white \
             --overlay $LH_OVERLAY \
             --rotate -o $OUTPUT_DIR/rotation.mp4
```

For more options see `whippersnap4 --help` or `whippersnap1 --help`.

## Quick Imports

```python
from whippersnappy import snap1, snap4, snap_rotate, plot3d
```

### Jupyter Notebooks:

WhipperSnapPy supports both static and **fully interactive 3D visualization** in Jupyter notebooks.

#### Interactive 3D Plotting

For **interactive mouse-controlled 3D rendering**:

```bash
pip install 'whippersnappy[notebook]'
```

```python
from whippersnappy import plot3d
from IPython.display import display

viewer = plot3d(
    meshpath='/path/to/surf/lh.white',
    curvpath='/path/to/surf/lh.curv',         # curvature
    overlaypath='/path/to/surf/lh.thickness', # optional: for colored overlays
    labelpath='/path/to/label/lh.cortex',     # optional: for masking
    minval=0.0,
    maxval=5.5,
    width=800,
    height=800,
)
display(viewer)
```

**Features:**
- ✅ Works in ALL Jupyter environments (browser, JupyterLab, Colab, VS Code)
- ✅ Mouse-controlled rotation, zoom, and pan
- ✅ Professional lighting (Three.js/WebGL)
- ✅ Supports overlays, annotations, and curvature
- ✅ Same technology Plotly uses for 3D plots

#### Static Rendering

For static publication-quality images:

```python
from whippersnappy import snap1
from whippersnappy.utils.types import ViewType
from IPython.display import display

img = snap1(
    meshpath='/path/to/surf/lh.white',
    overlaypath='/path/to/surf/lh.thickness',
    curvpath='/path/to/surf/lh.curv',
    view=ViewType.LEFT,  # or RIGHT, FRONT, BACK, TOP, BOTTOM
    width=800,
    height=800,
    brain_scale=1.5,
    specular=True,
)
display(img)
```

**Benefits:**
- ✅ Full PyOpenGL control for custom lighting
- ✅ Publication-quality output
- ✅ Fast performance
- ✅ Identical to GUI version

See `tutorials/whippersnappy_tutorial.ipynb` for complete examples.

### Desktop GUI:

For interactive desktop visualization with Qt controls:

```bash
whippersnap -lh /path/to/lh.thickness -sd /path/to/subject
```

This launches a native desktop GUI with a live OpenGL window and a
configuration panel for adjusting overlay thresholds at runtime.
Requires `pip install 'whippersnappy[gui]'`.

### Docker:

The Docker image provides a fully headless EGL rendering environment — no
display server or `xvfb` required.

Build the image:

```bash
docker build --rm -t whippersnappy -f Dockerfile .
```

Run a four-view batch snapshot:

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

For single-view snapshots, rotation videos, annotation overlays, custom
thresholds, and more examples see <a href="DOCKER.md"><strong>DOCKER.md</strong></a>.


## API Documentation

The API Documentation can be found at https://deep-mi.org/WhipperSnapPy .

## Links:

We also invite you to check out our lab webpage at https://deep-mi.org
