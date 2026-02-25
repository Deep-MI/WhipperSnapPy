"""Snapshot (static rendering) API for WhipperSnapPy."""

import logging
import os

import glfw
import numpy as np
import pyrr
from PIL import Image, ImageFont

from .geometry import estimate_overlay_thresholds, get_surf_name
from .geometry.prepare import prepare_and_validate_geometry
from .gl.utils import capture_window, create_window_with_fallback, render_scene, setup_shader, terminate_context
from .gl.views import get_view_matrices
from .utils.image import create_colorbar, draw_caption, draw_colorbar, load_roboto_font, text_size
from .utils.types import ColorSelection, OrientationType, ViewType

# Module logger
logger = logging.getLogger(__name__)


def snap1(
    mesh,
    outpath=None,
    overlay=None,
    annot=None,
    bg_map=None,
    roi=None,
    view=ViewType.LEFT,
    viewmat=None,
    width=700,
    height=500,
    fthresh=None,
    fmax=None,
    caption=None,
    caption_x=None,
    caption_y=None,
    caption_scale=1,
    invert=False,
    colorbar=True,
    colorbar_x=None,
    colorbar_y=None,
    colorbar_scale=1,
    orientation=OrientationType.HORIZONTAL,
    color_mode=ColorSelection.BOTH,
    font_file=None,
    specular=True,
    brain_scale=1.5,
    ambient=0.0,
):
    """Render a single static snapshot of a surface mesh.

    This function opens an OpenGL context, uploads the provided
    surface geometry and colors (overlay or annotation), renders the scene
    for a single view, captures the framebuffer, and returns a PIL Image.
    When ``outpath`` is provided the image is also written to disk.

    The mesh can be any triangular surface — not just brain surfaces.
    Supported file formats: FreeSurfer binary surface (e.g. ``lh.white``),
    ASCII OFF (``.off``), legacy ASCII VTK PolyData (``.vtk``), ASCII PLY
    (``.ply``), or a ``(vertices, faces)`` numpy array tuple.

    Parameters
    ----------
    mesh : str or tuple of (array-like, array-like)
        Path to a mesh file (FreeSurfer binary, ``.off``, ``.vtk``, or
        ``.ply``) **or** a ``(vertices, faces)`` tuple where *vertices* is
        (N, 3) float and *faces* is (M, 3) int.
    outpath : str or None, optional
        When provided, the resulting image is saved to this path.
    overlay : str, array-like, or None, optional
        Overlay file path (``.mgh`` or FreeSurfer morph) **or** a (N,) array
        of per-vertex scalar values.  If ``None``, coloring falls back to
        background shading / annotation.
    annot : str, tuple, or None, optional
        Path to a FreeSurfer .annot file **or** a ``(labels, ctab)`` /
        ``(labels, ctab, names)`` tuple with per-vertex labels.
    bg_map : str, array-like, or None, optional
        Path to a per-vertex scalar file **or** a (N,) array whose sign
        determines light/dark background shading for non-overlay vertices.
    roi : str, array-like, or None, optional
        Path to a FreeSurfer label file **or** a (N,) boolean array.
        Vertices with ``True`` receive overlay coloring; others fall back
        to *bg_map* shading.
    view : ViewType, optional
        Which pre-defined view to render (left, right, front, ...).
        Default is ``ViewType.LEFT``.
    viewmat : 4x4 matrix-like, optional
        Optional view matrix to override the pre-defined view.
    width, height : int, optional
        Output canvas size in pixels. Defaults to (700×500).
    fthresh, fmax : float or None, optional
        Threshold and saturation values for overlay coloring.
    caption, caption_x, caption_y, caption_scale : str/float, optional
        Caption text and layout parameters.
    invert : bool, optional
        Invert the color scale. Default is ``False``.
    colorbar : bool, optional
        If True, render a colorbar when an overlay is present. Default is ``True``.
    colorbar_x, colorbar_y, colorbar_scale : float, optional
        Colorbar positioning and scale. Scale defaults to 1.
    orientation : OrientationType, optional
        Colorbar orientation (HORIZONTAL/VERTICAL). Default is ``OrientationType.HORIZONTAL``.
    color_mode : ColorSelection, optional
        Which sign of overlay to color (POSITIVE/NEGATIVE/BOTH). Default is ``ColorSelection.BOTH``.
    font_file : str or None, optional
        Path to a TTF font for captions; fallback to bundled font if None.
    specular : bool, optional
        Enable specular highlights. Default is ``True``.
    brain_scale : float, optional
        Scale factor applied when preparing the geometry. Default is ``1.5``.
    ambient : float, optional
        Ambient lighting strength for shader. Default is ``0.0``.

    Returns
    -------
    PIL.Image.Image
        Rendered snapshot as a PIL Image.

    Raises
    ------
    RuntimeError
        If the OpenGL/GLFW context cannot be initialized.
    ValueError
        If the overlay contains no values to display for the chosen
        color_mode.
    FileNotFoundError
        If a required file cannot be found.

    Examples
    --------
    FreeSurfer surface with overlay::

        >>> from whippersnappy import snap1
        >>> img = snap1('lh.white', overlay='lh.thickness',
        ...             bg_map='lh.curv', roi='lh.cortex.label')
        >>> img.save('/tmp/lh.png')

    Array inputs (any triangular mesh)::

        >>> import numpy as np
        >>> v = np.random.randn(100, 3).astype(np.float32)
        >>> f = np.array([[0, 1, 2]], dtype=np.uint32)
        >>> img = snap1((v, f))

    OFF / VTK / PLY file::

        >>> img = snap1('mesh.off', overlay='values.mgh')
    """
    ref_width = 700
    ref_height = 500
    ui_scale = min(width / ref_width, height / ref_height)
    try:
        if glfw.init():
            primary_monitor = glfw.get_primary_monitor()
            if primary_monitor:
                mode = glfw.get_video_mode(primary_monitor)
                if width > mode.size.width:
                    logger.info("Requested width %d exceeds screen width %d, expect black bars",
                                width, mode.size.width)
                elif height > mode.size.height:
                    logger.info("Requested height %d exceeds screen height %d, expect black bars",
                                height, mode.size.height)
    except Exception:
        pass  # headless — no monitor info available, that's fine

    image = Image.new("RGB", (width, height))

    bwidth = int(540 * brain_scale * ui_scale)
    bheight = int(450 * brain_scale * ui_scale)
    brain_display_width = min(bwidth, width)
    brain_display_height = min(bheight, height)
    logger.debug("Requested (width,height) = (%s,%s)", width, height)
    logger.debug("Brain (width,height)     = (%s,%s)", bwidth, bheight)
    logger.debug("B-Display (width,height) = (%s,%s)", brain_display_width, brain_display_height)

    # will raise exception if it cannot be created
    window = create_window_with_fallback(brain_display_width, brain_display_height, "WhipperSnapPy", visible=True)
    try:
        meshdata, triangles, fthresh, fmax, pos, neg = prepare_and_validate_geometry(
            mesh,
            overlay,
            annot,
            bg_map,
            roi,
            fthresh,
            fmax,
            invert,
            scale=brain_scale,
            color_mode=color_mode,
        )

        shader = setup_shader(meshdata, triangles, brain_display_width, brain_display_height,
                              specular=specular, ambient=ambient)

        transl = pyrr.Matrix44.from_translation((0, 0, 0.4))
        view_mats = get_view_matrices()
        viewmat = transl * (view_mats[view] if viewmat is None else viewmat)
        render_scene(shader, triangles, viewmat)

        # Center the brain rendering in the output image, clamp to zero
        brain_x = max(0, (width - brain_display_width) // 2)
        brain_y = max(0, (height - brain_display_height) // 2)
        image.paste(capture_window(window), (brain_x, brain_y))

        bar = (
            create_colorbar(
                fthresh, fmax, invert, orientation, colorbar_scale * ui_scale, pos, neg, font_file=font_file
            )
            if overlay is not None and colorbar
            else None
        )
        font = (
            load_roboto_font(int(20 * caption_scale * ui_scale))
            if font_file is None
            else ImageFont.truetype(font_file, int(20 * caption_scale * ui_scale))
            if caption
            else None
        )

        # Compute positions to avoid overlap, unless explicit positions are given
        text_w, text_h = text_size(caption, font) if caption and font else (0, 0)
        bar_h = bar.height if bar is not None else 0
        gap = int(4 * ui_scale)
        bottom_pad = int(20 * ui_scale)

        if orientation == OrientationType.HORIZONTAL:
            # If explicit positions are given, use them
            if colorbar_x is not None or colorbar_y is not None or caption_x is not None or caption_y is not None:
                bx = int(colorbar_x * width) if colorbar_x is not None else None
                by = int(colorbar_y * height) if colorbar_y is not None else None
                cx = int(caption_x * width) if caption_x is not None else None
                cy = int(caption_y * height) if caption_y is not None else None
                draw_colorbar(image, bar, orientation, x=bx, y=by)
                draw_caption(image, caption, font, orientation, x=cx, y=cy)
            else:
                # Place colorbar above caption if both present
                if bar is not None and caption:
                    bar_y = image.height - bottom_pad - text_h - gap - bar_h
                    caption_y = image.height - bottom_pad - text_h
                elif bar is not None:
                    bar_y = image.height - bottom_pad - bar_h
                    caption_y = None
                elif caption:
                    bar_y = None
                    caption_y = image.height - bottom_pad - text_h
                else:
                    bar_y = caption_y = None
                draw_colorbar(image, bar, orientation, y=bar_y)
                draw_caption(image, caption, font, orientation, y=caption_y)
        else:
            # For vertical, allow explicit x/y for both, else use default
            bx = int(colorbar_x * width) if colorbar_x is not None else None
            by = int(colorbar_y * height) if colorbar_y is not None else None
            cx = int(caption_x * width) if caption_x is not None else None
            cy = int(caption_y * height) if caption_y is not None else None
            draw_colorbar(image, bar, orientation, x=bx, y=by)
            draw_caption(image, caption, font, orientation, x=cx, y=cy)

        if outpath:
            logger.info("Saving snapshot to %s", outpath)
            image.save(outpath)
        return image
    finally:
        terminate_context(window)


def snap4(
    lh_overlay=None,
    rh_overlay=None,
    lh_annot=None,
    rh_annot=None,
    fthresh=None,
    fmax=None,
    sdir=None,
    caption=None,
    invert=False,
    roi_name="cortex.label",
    surfname=None,
    bg_map_name="curv",
    colorbar=True,
    outpath=None,
    font_file=None,
    specular=True,
    ambient=0.0,
    brain_scale=1.85,
    color_mode=ColorSelection.BOTH,
):
    """Render four snapshot views (left/right hemispheres, lateral/medial).

    This convenience function renders four views (lateral/medial for each
    hemisphere), stitches them together into a single PIL Image and returns
    it (and saves it to ``outpath`` when provided). It is typically used to
    produce publication-ready overview figures composed from both
    hemispheres.

    Parameters
    ----------
    lh_overlay, rh_overlay : str, array-like, or None
        Left/right hemisphere overlay — either a file path (FreeSurfer morph
        or .mgh) or a per-vertex scalar array.  Typically provided as a pair
        for a coherent two-hemisphere color scale.
    lh_annot, rh_annot : str, tuple, or None
        Left/right hemisphere annotation — either a path to a .annot file or
        a ``(labels, ctab)`` / ``(labels, ctab, names)`` tuple.
        Cannot be combined with ``lh_overlay``/``rh_overlay``.
    fthresh, fmax : float or None
        Threshold and saturation for overlay coloring.  Auto-estimated when
        ``None``.
    sdir : str or None
        Subject directory containing ``surf/`` and ``label/`` subdirectories.
        Falls back to ``$SUBJECTS_DIR`` when ``None``.
    caption : str or None
        Caption string to place on the final image.
    invert : bool, optional
        Invert color scale. Default is ``False``.
    roi_name : str, optional
        Basename of the label file used to restrict overlay coloring (default
        ``'cortex.label'``).  The full path is constructed as
        ``<sdir>/label/<hemi>.<roi_name>``.
    surfname : str or None, optional
        Surface basename to load (e.g. ``'white'``); auto-detected when
        ``None``.
    bg_map_name : str, optional
        Basename of the curvature/morph file used for background shading
        (default ``'curv'``).  The full path is constructed as
        ``<sdir>/surf/<hemi>.<bg_map_name>``.
    colorbar : bool, optional
        Whether to draw a colorbar on the composed image. Default is ``True``.
    outpath : str or None, optional
        If provided, save composed image to this path.
    font_file : str or None, optional
        Path to a font to use for captions.
    specular : bool, optional
        Enable/disable specular highlights in the renderer. Default is ``True``.
    ambient : float, optional
        Ambient lighting strength. Default is ``0``.
    brain_scale : float, optional
        Scaling factor passed to geometry preparation. Default is ``1.85``.
    color_mode : ColorSelection, optional
        Which sign of overlay to color (POSITIVE/NEGATIVE/BOTH). Default is ``ColorSelection.BOTH``.

    Returns
    -------
    PIL.Image.Image
        Composed image of the four views.

    Raises
    ------
    ValueError
        For invalid argument combinations or when required overlay values
        are absent.
    FileNotFoundError
        When required surface files are not found.

    Examples
    --------
    >>> from whippersnappy import snap4
    >>> img = snap4(
    ...     lh_overlay='fsaverage/surf/lh.thickness',
    ...     rh_overlay='fsaverage/surf/rh.thickness',
    ...     sdir='./fsaverage'
    ... )
    >>> img.save('/tmp/whippersnappy_overview.png')
    """
    wwidth = 540
    wheight = 450

    # Resolve sdir early so path-building works for both the pre-pass and
    # the rendering loop.
    if sdir is None:
        sdir = os.environ.get("SUBJECTS_DIR")
        if not sdir and surfname is None:
            logger.error("No sdir or SUBJECTS_DIR provided")
            raise ValueError("No sdir or SUBJECTS_DIR provided")
        if not sdir and surfname is not None:
            logger.error("surfname provided but sdir is None")
            raise ValueError("surfname provided but sdir is None; cannot construct mesh path.")

    # Pre-pass: estimate missing fthresh/fmax from overlays for global color scale
    has_overlay = lh_overlay is not None or rh_overlay is not None
    if has_overlay and (fthresh is None or fmax is None):
        est_fthreshs = []
        est_fmaxs = []
        for _overlay in filter(None, (lh_overlay, rh_overlay)):
            h_fthresh, h_fmax = estimate_overlay_thresholds(_overlay, fthresh, fmax)
            est_fthreshs.append(h_fthresh)
            est_fmaxs.append(h_fmax)
        if fthresh is None and est_fthreshs:
            fthresh = min(est_fthreshs)
        if fmax is None and est_fmaxs:
            fmax = max(est_fmaxs)
        logger.debug("Global color range: fthresh=%s  fmax=%s", fthresh, fmax)

    # will raise exception if it cannot be created
    window = create_window_with_fallback(wwidth, wheight, "WhipperSnapPy", visible=True)
    try:
        # Use standard view matrices from get_view_matrices and ViewType
        view_mats = get_view_matrices()
        view_left = view_mats[ViewType.LEFT]
        view_right = view_mats[ViewType.RIGHT]
        transl = pyrr.Matrix44.from_translation((0, 0, 0.4))

        # Predefine hemisphere images so static analysis knows they exist even if
        # an earlier step raises an exception (we still will fail at runtime).
        lhimg = None
        rhimg = None

        for hemi in ("lh", "rh"):
            if surfname is None:
                found_surfname = get_surf_name(sdir, hemi)
                if found_surfname is None:
                    logger.error("Could not find valid surface in %s for hemi: %s!", sdir, hemi)
                    raise FileNotFoundError(f"Could not find valid surface in {sdir} for hemi: {hemi}")
                mesh = os.path.join(sdir, "surf", hemi + "." + found_surfname)
            else:
                mesh = os.path.join(sdir, "surf", hemi + "." + surfname)

            # Assign derived paths for bg_map and roi
            bg_map = os.path.join(sdir, "surf", hemi + "." + bg_map_name) if bg_map_name else None
            roi = os.path.join(sdir, "label", hemi + "." + roi_name) if roi_name else None
            overlay = lh_overlay if hemi == "lh" else rh_overlay
            annot = lh_annot if hemi == "lh" else rh_annot

            # If overlay is an array, it doesn't have a path to log; handle gracefully
            if isinstance(overlay, str):
                logger.debug("overlay=%s exists=%s", overlay, os.path.exists(overlay))
            elif overlay is not None:
                logger.debug("overlay=<array shape=%s>", getattr(overlay, 'shape', None))

            # Diagnostic: report mesh and overlay paths and whether they exist
            logger.debug("hemisphere=%s", hemi)
            if isinstance(mesh, str):
                logger.debug("mesh=%s exists=%s", mesh, os.path.exists(mesh))
            if isinstance(annot, str) and annot is not None:
                logger.debug("annot=%s exists=%s", annot, os.path.exists(annot))
            if bg_map is not None:
                logger.debug("bg_map=%s exists=%s", bg_map, os.path.exists(bg_map))

            try:
                meshdata, triangles, fthresh, fmax, pos, neg = prepare_and_validate_geometry(
                    mesh, overlay, annot, bg_map, roi, fthresh, fmax, invert,
                    scale=brain_scale, color_mode=color_mode
                )
            except Exception as e:
                logger.error("prepare_geometry failed for %s: %s", mesh, e)
                raise

            # Diagnostics about mesh data
            try:
                logger.debug("meshdata shape: %s; triangles count: %s", getattr(meshdata, 'shape', None),
                             getattr(triangles, 'size', None))
            except Exception:
                pass

            try:
                shader = setup_shader(meshdata, triangles, wwidth, wheight, specular=specular, ambient=ambient)
                logger.debug("Shader setup complete")
            except Exception as e:
                logger.error("setup_shader failed: %s", e)
                raise

            render_scene(shader, triangles, transl * view_left)
            im1 = capture_window(window)
            render_scene(shader, triangles, transl * view_right)
            im2 = capture_window(window)

            if hemi == "lh":
                lhimg = Image.new("RGB", (im1.width, im1.height + im2.height))
                lhimg.paste(im1, (0, 0))
                lhimg.paste(im2, (0, im1.height))
            else:
                rhimg = Image.new("RGB", (im1.width, im1.height + im2.height))
                # For right hemisphere, reverse the order: top=im2, bottom=im1
                rhimg.paste(im2, (0, 0))
                rhimg.paste(im1, (0, im2.height))

        # Add small padding around each hemisphere to avoid cropping at edges
        pad = max(4, int(0.03 * wwidth))
        padded_lh = Image.new("RGB", (lhimg.width + 2 * pad, lhimg.height + 2 * pad), (0, 0, 0))
        padded_lh.paste(lhimg, (pad, pad))
        padded_rh = Image.new("RGB", (rhimg.width + 2 * pad, rhimg.height + 2 * pad), (0, 0, 0))
        padded_rh.paste(rhimg, (pad, pad))

        image = Image.new("RGB", (padded_lh.width + padded_rh.width, padded_lh.height))
        image.paste(padded_lh, (0, 0))
        image.paste(padded_rh, (padded_lh.width, 0))

        font = load_roboto_font(20) if font_file is None else ImageFont.truetype(font_file, 20) if caption else None
        # Place caption at bottom, colorbar above if both present
        text_w, text_h = text_size(caption, font) if caption and font else (0, 0)
        bottom_pad = 20
        gap = 4
        caption_y = image.height - bottom_pad - text_h
        bar = (
            create_colorbar(fthresh, fmax, invert, pos=pos, neg=neg)
            if lh_annot is None and rh_annot is None and colorbar
            else None
        )
        bar_h = bar.height if bar is not None else 0
        if bar is not None and caption:
            bar_y = image.height - bottom_pad - text_h - gap - bar_h
            draw_colorbar(image, bar, OrientationType.HORIZONTAL, y=bar_y)
            draw_caption(image, caption, font, OrientationType.HORIZONTAL, y=caption_y)
        elif bar is not None:
            bar_y = image.height - bottom_pad - bar_h
            draw_colorbar(image, bar, OrientationType.HORIZONTAL, y=bar_y)
        elif caption:
            draw_caption(image, caption, font, OrientationType.HORIZONTAL, y=caption_y)

        # If outpath is specified, save to disk
        if outpath:
            logger.info("Saving snapshot to %s", outpath)
            image.save(outpath)

        return image
    finally:
        terminate_context(window)


def snap_rotate(
    mesh,
    outpath,
    n_frames=72,
    fps=24,
    width=700,
    height=500,
    overlay=None,
    bg_map=None,
    annot=None,
    roi=None,
    fthresh=None,
    fmax=None,
    invert=False,
    specular=True,
    ambient=0.0,
    brain_scale=1.5,
    start_view=ViewType.LEFT,
    color_mode=ColorSelection.BOTH,
):
    """Render a rotating 360° video of a surface mesh.

    Rotates the view around the vertical (Y) axis in ``n_frames`` equal
    steps, captures each frame via OpenGL, and encodes the result into a
    video file.  An animated GIF can be produced by passing an ``outpath``
    ending in ``.gif``; in that case ``imageio-ffmpeg`` is not required.

    The mesh can be any triangular surface — not just brain surfaces.
    Supported file formats: FreeSurfer binary surface, ASCII OFF (``.off``),
    legacy ASCII VTK PolyData (``.vtk``), ASCII PLY (``.ply``), or a
    ``(vertices, faces)`` numpy array tuple.

    Parameters
    ----------
    mesh : str or tuple of (array-like, array-like)
        Path to a mesh file (FreeSurfer binary, ``.off``, ``.vtk``, or
        ``.ply``) **or** a ``(vertices, faces)`` tuple.
    outpath : str
        Destination file path.  The extension controls the output format:

        * ``.mp4`` — H.264 MP4 (recommended, requires ``imageio-ffmpeg``).
        * ``.webm`` — VP9 WebM (requires ``imageio-ffmpeg``).
        * ``.gif`` — animated GIF (no ffmpeg required, but larger file).

    n_frames : int, optional
        Number of frames for a full 360° rotation. Default is ``72``
        (one frame every 5°).
    fps : int, optional
        Output frame rate in frames per second. Default is ``24``.
    width, height : int, optional
        Render resolution in pixels. Defaults are ``700`` and ``500``.
    overlay : str, array-like, or None, optional
        Per-vertex overlay file path or array (e.g. thickness).
    bg_map : str, array-like, or None, optional
        Curvature/morph file path or array for background shading.
    annot : str, tuple, or None, optional
        FreeSurfer ``.annot`` file path or ``(labels, ctab)`` tuple.
    roi : str, array-like, or None, optional
        Label file path or boolean array to restrict overlay coloring.
    fthresh : float or None, optional
        Overlay threshold value.
    fmax : float or None, optional
        Overlay saturation value.
    invert : bool, optional
        Invert the overlay color scale. Default is ``False``.
    specular : bool, optional
        Enable specular highlights. Default is ``True``.
    ambient : float, optional
        Ambient lighting strength. Default is ``0.0``.
    brain_scale : float, optional
        Geometry scale factor. Default is ``1.5``.
    start_view : ViewType, optional
        Pre-defined view to start the rotation from.
        Default is ``ViewType.LEFT``.
    color_mode : ColorSelection, optional
        Which overlay sign to color (POSITIVE/NEGATIVE/BOTH).
        Default is ``ColorSelection.BOTH``.

    Returns
    -------
    str
        The resolved ``outpath`` that was written.

    Raises
    ------
    ImportError
        If ``imageio`` or ``imageio-ffmpeg`` is not installed and a
        video format (``.mp4``, ``.webm``) was requested.
    RuntimeError
        If the OpenGL context cannot be initialised.
    ValueError
        If the overlay contains no values for the chosen color mode.

    Examples
    --------
    >>> from whippersnappy import snap_rotate
    >>> snap_rotate(
    ...     'fsaverage/surf/lh.white',
    ...     '/tmp/rotation.mp4',
    ...     overlay='fsaverage/surf/lh.thickness',
    ... )
    '/tmp/rotation.mp4'
    """
    ext = os.path.splitext(outpath)[1].lower()
    use_gif = ext == ".gif"

    if not use_gif:
        try:
            import imageio  # noqa: F401
            import imageio_ffmpeg  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"Video output requires the 'imageio' and 'imageio-ffmpeg' packages. "
                f"Install with: pip install 'whippersnappy[video]'\n"
                f"Original error: {exc}"
            ) from exc
        import imageio
    else:
        try:
            import imageio  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "GIF output requires the 'imageio' package. "
                "Install with: pip install 'whippersnappy[video]'"
            ) from exc
        import imageio

    window = create_window_with_fallback(width, height, "WhipperSnapPy", visible=True)
    try:
        meshdata, triangles, fthresh, fmax, pos, neg = prepare_and_validate_geometry(
            mesh,
            overlay,
            annot,
            bg_map,
            roi,
            fthresh,
            fmax,
            invert,
            scale=brain_scale,
            color_mode=color_mode,
        )
        logger.info(
            "Rendering %d frames at %dx%d (%.0f° per step) → %s",
            n_frames, width, height, 360.0 / n_frames, outpath,
        )

        shader = setup_shader(meshdata, triangles, width, height,
                              specular=specular, ambient=ambient)

        transl = pyrr.Matrix44.from_translation((0, 0, 0.4))
        base_view = get_view_matrices()[start_view]

        frames = []
        for i in range(n_frames):
            angle = 2 * np.pi * i / n_frames
            rot = pyrr.Matrix44.from_y_rotation(angle)
            viewmat = transl * rot * base_view
            render_scene(shader, triangles, viewmat)
            frames.append(np.array(capture_window(window)))
            if (i + 1) % max(1, n_frames // 10) == 0:
                logger.debug("  frame %d / %d", i + 1, n_frames)

    finally:
        terminate_context(window)

    logger.info("Encoding %d frames to %s …", len(frames), outpath)
    if use_gif:
        # Pure-PIL GIF — no ffmpeg required
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            outpath,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=int(1000 / fps),
            optimize=True,
        )
    else:
        writer_kwargs = {
            "fps": fps,
            "codec": "libx264",
            "quality": 6,
            "pixelformat": "yuv420p",
        }
        if ext == ".webm":
            writer_kwargs["codec"] = "libvpx-vp9"
            writer_kwargs.pop("pixelformat", None)
        imageio.mimwrite(outpath, frames, **writer_kwargs)

    logger.info("Saved rotation video to %s", outpath)
    return outpath

