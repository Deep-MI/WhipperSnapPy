"""Image and text helper utilities used by snapshot renderers (moved under utils).
"""
import numpy as np
from PIL import Image, ImageDraw

from whippersnappy.utils.colormap import heat_color
from whippersnappy.utils.types import OrientationType

try:
    # Prefer stdlib importlib.resources
    from importlib import resources
except Exception:
    import importlib_resources as resources
import warnings

from PIL import ImageFont


def text_size(caption, font):
    """Return text width and height in pixels."""
    dummy_img = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), caption, font=font, anchor="lt")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height


def get_colorbar_label_positions(
    font,
    labels,
    colorbar_rect,
    gapspace=0,
    pos=True,
    neg=True,
    orientation=OrientationType.HORIZONTAL,
):
    """Return label positions for a colorbar."""
    positions = {}
    cb_x, cb_y, cb_width, cb_height = colorbar_rect
    cb_labels_gap = 5

    if orientation == OrientationType.HORIZONTAL:
        label_y = cb_y + cb_height + cb_labels_gap

        w, _ = text_size(labels["upper"], font)
        if pos:
            positions["upper"] = (cb_x + cb_width - w, label_y)
        else:
            upper_x = cb_x + cb_width - w - int(gapspace) if gapspace > 0 else cb_x + cb_width - w
            positions["upper"] = (upper_x, label_y)

        w, _ = text_size(labels["lower"], font)
        if neg:
            positions["lower"] = (cb_x, label_y)
        else:
            lower_x = cb_x + int(gapspace) if gapspace > 0 else cb_x
            positions["lower"] = (lower_x, label_y)

        if neg and pos:
            if gapspace == 0:
                w, _ = text_size(labels["middle"], font)
                positions["middle"] = (cb_x + cb_width // 2 - w // 2, label_y)
            else:
                w, _ = text_size(labels["middle_neg"], font)
                positions["middle_neg"] = (cb_x + cb_width // 2 - w - int(gapspace), label_y)
                w, _ = text_size(labels["middle_pos"], font)
                positions["middle_pos"] = (cb_x + cb_width // 2 + int(gapspace), label_y)
    else:
        label_x = cb_x + cb_width + cb_labels_gap

        _, h = text_size(labels["upper"], font)
        if pos:
            positions["upper"] = (label_x, cb_y)
        else:
            upper_y = cb_y + int(gapspace) if gapspace > 0 else cb_y
            positions["upper"] = (label_x, upper_y)

        _, h = text_size(labels["lower"], font)
        if neg:
            positions["lower"] = (label_x, cb_y + cb_height - 1.5 * h)
        else:
            lower_y = cb_y + cb_height - int(gapspace) - 1.5 * h if gapspace > 0 else cb_y + cb_height - 1.5 * h
            positions["lower"] = (label_x, lower_y)

        if neg and pos:
            if gapspace == 0:
                _, h = text_size(labels["middle"], font)
                positions["middle"] = (label_x, cb_y + cb_height // 2 - h // 2)
            else:
                _, h = text_size(labels["middle_pos"], font)
                positions["middle_pos"] = (label_x, cb_y + cb_height // 2 - 1.5 * h - int(gapspace))
                _, h = text_size(labels["middle_neg"], font)
                positions["middle_neg"] = (label_x, cb_y + cb_height // 2 + int(gapspace))

    return positions


def create_colorbar(
    fmin,
    fmax,
    invert,
    orientation=OrientationType.HORIZONTAL,
    colorbar_scale=1,
    pos=True,
    neg=True,
    font_file=None,
):
    """Create a colorbar image (PIL.Image) using the project's heat_color.

    Parameters mirror the previous implementation in `render.py`.
    """
    # If fmin/fmax are not specified, we cannot create a meaningful colorbar.
    if fmin is None or fmax is None:
        return None

    cwidth = int(200 * colorbar_scale)
    cheight = int(30 * colorbar_scale)
    gapspace = 0

    if fmin > 0.01:
        num = int(0.42 * cwidth)
        gapspace = 0.08 * cwidth
    else:
        num = int(0.5 * cwidth)
    if not neg or not pos:
        num = num * 2
        gapspace = gapspace * 2

    values = np.nan * np.ones(cwidth)
    steps = np.linspace(0.01, 1, num)
    if pos and not neg:
        values[-steps.size:] = steps
    elif not pos and neg:
        values[: steps.size] = -1.0 * np.flip(steps)
    else:
        values[: steps.size] = -1.0 * np.flip(steps)
        values[-steps.size:] = steps

    colors = heat_color(values, invert)
    colors[np.isnan(values), :] = 0.33 * np.ones((1, 3))
    img_bar = np.uint8(np.tile(colors, (cheight, 1, 1)) * 255)

    pad_top, pad_left = 3, 10
    img_buf = np.zeros((cheight + 2 * pad_top, cwidth + 2 * pad_left, 3), dtype=np.uint8)
    img_buf[pad_top : cheight + pad_top, pad_left : cwidth + pad_left, :] = img_bar
    image = Image.fromarray(img_buf)

    if font_file is None:
        # Try to load bundled font from package resources
        font = None
        try:
            font_trav = resources.files("whippersnappy").joinpath("resources", "fonts", "Roboto-Regular.ttf")
            with resources.as_file(font_trav) as font_path:
                font = ImageFont.truetype(str(font_path), int(12 * colorbar_scale))
        except Exception:
            warnings.warn("Roboto font not found in package resources; falling back to default font", UserWarning)
            font = ImageFont.load_default()
    else:
        try:
            font = ImageFont.truetype(font_file, int(12 * colorbar_scale))
        except Exception:
            font = ImageFont.load_default()

    labels = {}
    labels["upper"] = f">{fmax:.2f}" if pos else (f"{-fmin:.2f}" if gapspace != 0 else "0")
    labels["lower"] = f"<{-fmax:.2f}" if neg else (f"{fmin:.2f}" if gapspace != 0 else "0")
    if neg and pos and gapspace != 0:
        labels["middle_neg"] = f"{-fmin:.2f}"
        labels["middle_pos"] = f"{fmin:.2f}"
    elif neg and pos and gapspace == 0:
        labels["middle"] = "0"

    caption_sizes = [text_size(caption, font) for caption in labels.values()]
    max_caption_width = int(max([caption_size[0] for caption_size in caption_sizes]))
    max_caption_height = int(max([caption_size[1] for caption_size in caption_sizes]))

    if orientation == OrientationType.VERTICAL:
        image = image.rotate(90, expand=True)
        new_width = image.width + int(max_caption_width)
        new_image = Image.new("RGB", (new_width, image.height), (0, 0, 0))
        new_image.paste(image, (0, 0))
        image = new_image
        colorbar_rect = (pad_top, pad_left, cheight, cwidth)
    else:
        new_height = image.height + int(max_caption_height * 2)
        new_image = Image.new("RGB", (image.width, new_height), (0, 0, 0))
        new_image.paste(image, (0, 0))
        image = new_image
        colorbar_rect = (pad_left, pad_top, cwidth, cheight)

    positions = get_colorbar_label_positions(font, labels, colorbar_rect, gapspace, pos, neg, orientation)
    draw = ImageDraw.Draw(image)
    for label_key, position in positions.items():
        draw.text((int(position[0]), int(position[1])), labels[label_key], fill=(220, 220, 220), font=font)

    return image


def load_roboto_font(size=14):
    """Load bundled Roboto-Regular.ttf from package resources.

    Returns a PIL ImageFont instance. Falls back to ImageFont.load_default()
    if the bundled font isn't available.
    """
    try:
        # resources was imported earlier in this module
        font_trav = resources.files("whippersnappy").joinpath("resources", "fonts", "Roboto-Regular.ttf")
        with resources.as_file(font_trav) as font_path:
            return ImageFont.truetype(str(font_path), size)
    except Exception:
        warnings.warn("Roboto font not found in package resources; falling back to default font", UserWarning)
        try:
            return ImageFont.load_default()
        except Exception:
            return None

