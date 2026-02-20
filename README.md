# WhipperSnapPy

WhipperSnapPy is a Python/OpenGL tool to render FreeSurfer and FastSurfer
surface models with color overlays or parcellations and generate screenshots
— from the command line, in Jupyter notebooks, or via a desktop GUI.

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

Renders one surface view:

```bash
whippersnap1 $SUBJECT_DIR/surf/lh.white \
             --overlay $LH_OVERLAY \
             --view left \
             -o snap1.png
```

### Rotation video (`whippersnap1 --rotate`)

Renders a 360° animation:

```bash
whippersnap1 $SUBJECT_DIR/surf/lh.white \
             --overlay $LH_OVERLAY \
             --rotate \
             -o rotation.mp4
```

### Desktop GUI (`whippersnap`)

Launches an interactive Qt window with live threshold controls:

```bash
pip install 'whippersnappy[gui]'
whippersnap --lh $LH_OVERLAY --sdir $SUBJECT_DIR
```

For all options run `whippersnap4 --help`, `whippersnap1 --help`, or `whippersnap --help`.

## Python API

```python
from whippersnappy import snap1, snap4, snap_rotate, plot3d
```

| Function | Description |
|---|---|
| `snap1` | Single-view surface snapshot → PIL Image |
| `snap4` | Four-view composed image (lateral/medial, both hemispheres) |
| `snap_rotate` | 360° rotation video (MP4, WebM, or GIF) |
| `plot3d` | Interactive 3D WebGL viewer for Jupyter notebooks |

### Example

```python
from whippersnappy import snap4
img = snap4(sdir='/path/to/subject',
            lhoverlaypath='/path/to/lh.thickness',
            rhoverlaypath='/path/to/rh.thickness',
            colorbar=True, caption='Cortical Thickness (mm)')
img.save('snap4.png')
```

See `tutorials/whippersnappy_tutorial.ipynb` for complete notebook examples.

## Docker

The Docker image provides a fully headless EGL rendering environment — no
display server or `xvfb` required. See <a href="DOCKER.md"><strong>DOCKER.md</strong></a> for details.

## API Documentation

https://deep-mi.org/WhipperSnapPy

## Links

Lab webpage: https://deep-mi.org
