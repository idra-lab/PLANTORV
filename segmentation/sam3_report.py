"""Debug dumps and the HTML mask viewer of SAM 3 segmentations.

Presentation only: nothing here changes what :class:`segmentation.sam3_model.SAM3Model`
segments. :func:`dump_masks` writes the masks of one prediction for inspection, and
:func:`render_view` builds the self-contained page ``SAM3Model.visualize`` writes.
"""

import base64
import html
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from utility.masks import binary_mask_to_pil
from utility.utility import logger

# Thickness of the mask outlines in the debug overlay.
OUTLINE_WIDTH = 3

# Longest side, in pixels, of the images embedded in the HTML view. The page
# carries them base64-encoded so that it stays a single self-contained file that
# can be copied off a cluster node, which makes their size the page's size.
VIEW_OVERLAY_MAX = 1600
VIEW_CROP_MAX = 360

# Styling for the page written by :func:`render_view`. Kept out of the builder
# so that the braces of the CSS do not have to be escaped inside an f-string.
VIEWER_CSS = """
body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 0; padding: 24px; background: #14161a; color: #e6e8ec; }
h1 { font-size: 19px; margin: 0 0 8px; font-weight: 600; }
p.meta { color: #9aa1ac; font-size: 13px; margin: 0 0 4px; }
p.meta b { color: #e6e8ec; font-weight: 600; }
p.missing b { color: #f0a868; }
img.overlay { width: 100%; border-radius: 8px; display: block; margin: 20px 0; }
.grid { display: grid; gap: 16px;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
figure { margin: 0; background: #1c1f25; border: 1px solid #2a2f38;
         border-radius: 8px; padding: 10px; }
figure img { width: 100%; display: block; background: #000; border-radius: 4px; }
figcaption { font-size: 12px; margin-top: 8px; }
figcaption b { display: block; font-size: 13px; margin-bottom: 2px; }
figcaption span { color: #9aa1ac; }
.swatch { display: inline-block; width: 9px; height: 9px;
          border-radius: 2px; margin-right: 6px; }
"""


def slug(label: str) -> str:
    """Return a concept label usable in a file name.

    Parameters
    ----------
    label : str
        The concept.

    Returns
    -------
    str
        The label with everything but letters and digits replaced by
        underscores, so that a phrase containing a slash or a space cannot
        turn into a nested path.
    """
    return "".join(c if c.isalnum() else "_" for c in label)


def colours(count: int) -> List[np.ndarray]:
    """Return one distinct RGB colour per mask.

    Parameters
    ----------
    count : int
        How many colours are needed.

    Returns
    -------
    list[numpy.ndarray]
        Colours spread over the hue circle in fixed steps, which keeps
        neighbouring masks visually distinct.
    """
    return [
        cv2.cvtColor(
            np.array([[[int(180 * i / max(count, 1)), 255, 255]]], dtype=np.uint8),
            cv2.COLOR_HSV2RGB,
        )[0, 0].astype(np.uint16)
        for i in range(count)
    ]


def draw_overlay(image: np.ndarray, masks: np.ndarray, labels: List[str]) -> np.ndarray:
    """Draw every mask on the image as a coloured outline with its concept.

    Outlines only, never a fill. Colour is what the concept phrases are built
    on, and a fill repaints the objects in a hue picked per mask index rather
    than per object, which makes the one question worth asking of the picture
    - did "yellow structure" land on something yellow? - unanswerable.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image with shape ``(H, W, 3)``.
    masks : numpy.ndarray
        Boolean masks with shape ``(N, H, W)``.
    labels : list[str]
        The concept each mask was found by.

    Returns
    -------
    numpy.ndarray
        A copy of ``image`` with the outlines and labels drawn on it.
    """
    overlay = image.copy()
    colours_ = colours(len(masks))

    for mask, colour in zip(masks, colours_):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, colour.tolist(), OUTLINE_WIDTH)

    # Labels go on after every outline, so that an outline drawn later cannot
    # run through the text of a mask drawn earlier.
    for i, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        cv2.putText(
            overlay,
            f"{i}:{labels[i]}",
            (int(xs.mean()), int(ys.mean())),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def dump_masks(
    save_dir: Path,
    image: np.ndarray,
    masks: np.ndarray,
    bboxes: List[List[int]],
    labels: List[str],
    tag: str,
) -> None:
    """Write the masks of one prediction to ``save_dir/debug/`` for inspection.

    Produces one colour-coded overlay of all masks at once, one binary PNG per
    mask named after the concept that found it, and a log line per mask, so a
    concept the VLM never named can be told apart from one SAM 3 could not
    find.

    Parameters
    ----------
    save_dir : Path
        The model's ``save_dir``; the dumps go under its ``debug/``.
    image : numpy.ndarray
        RGB image the masks were generated from, shape ``(H, W, 3)``.
    masks : numpy.ndarray
        Boolean masks with shape ``(N, H, W)``.
    bboxes : list[list[int]]
        One box per mask as ``[x_min, y_min, width, height]``.
    labels : list[str]
        The concept each mask was found by.
    tag : str
        Sub-directory name, e.g. ``"image_1"``.
    """
    debug_dir = Path(save_dir) / "debug" / tag
    debug_dir.mkdir(parents=True, exist_ok=True)

    H, W = image.shape[:2]

    logger.debug(f"[{tag}] SAM 3 returned {len(masks)} masks")
    for i, mask in enumerate(masks):
        area_pct = np.sum(mask) * 100 / (H * W)
        logger.debug(
            f"[{tag}]   mask {i:03d}: {labels[i]!r} area {area_pct:6.2f}% bbox {bboxes[i]}"
        )
        binary_mask_to_pil(mask.astype(np.uint8)).save(
            debug_dir / f"mask_{i:03d}_{slug(labels[i])}.png"
        )

    Image.fromarray(draw_overlay(image, masks, labels)).save(
        debug_dir.parent / f"{tag}_all_masks.png"
    )
    logger.debug(f"[{tag}] wrote overlay and {len(masks)} masks to {debug_dir}")


def data_uri(array: np.ndarray, max_side: int) -> str:
    """Encode an image as a PNG ``data:`` URI, downscaled to fit.

    Parameters
    ----------
    array : numpy.ndarray
        RGB image with shape ``(H, W, 3)``.
    max_side : int
        Longest side the image is shrunk to before encoding.

    Returns
    -------
    str
        The image as ``data:image/png;base64,...``.
    """
    img = Image.fromarray(array.astype(np.uint8))
    img.thumbnail((max_side, max_side))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_view(
    image: np.ndarray,
    masks: np.ndarray,
    bboxes: List[List[int]],
    labels: List[str],
    idx: int,
    *,
    concepts: List[str],
    task: Optional[str],
    task_relevant: List[str],
) -> str:
    """Build the self-contained HTML page showing one segmentation.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image the masks were produced from, shape ``(H, W, 3)``.
    masks : numpy.ndarray
        Boolean masks with shape ``(N, H, W)``.
    bboxes : list[list[int]]
        One box per mask as ``[x_min, y_min, width, height]``.
    labels : list[str]
        The concept each mask was found by.
    idx : int
        Image index, shown in the heading.
    concepts : list[str]
        The phrases SAM 3 was prompted with.
    task : str or None
        The configured task, if any.
    task_relevant : list[str]
        The phrases the VLM considers necessary for the task.

    Returns
    -------
    str
        The complete HTML document.
    """
    height, width = image.shape[:2]
    colours_ = colours(len(masks))

    header = [f"<h1>image {idx + 1} &mdash; {len(masks)} masks from {len(concepts)} concepts</h1>"]
    if task:
        header.append(f"<p class='meta'>task: <b>{html.escape(task)}</b></p>")
    if task_relevant:
        relevant = ", ".join(html.escape(c) for c in task_relevant)
        header.append(f"<p class='meta'>task-relevant: <b>{relevant}</b></p>")

    # A phrase that matched nothing is invisible in the picture, and is
    # exactly what a badly chosen concept looks like, so it is named here.
    missing = [c for c in concepts if c not in set(labels)]
    if missing:
        names = ", ".join(html.escape(c) for c in missing)
        header.append(f"<p class='meta missing'>found nothing for: <b>{names}</b></p>")

    cards = []
    for i, (mask, colour) in enumerate(zip(masks, colours_)):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1

        # Deliberately not `SAM3Model._crop`: that boosts contrast for the annotator,
        # which shifts the colours this page exists to let you check.
        cut = image[top:bottom, left:right] * mask[top:bottom, left:right, None]

        swatch = "#{:02x}{:02x}{:02x}".format(*(int(c) for c in colour))
        area = mask.sum() * 100 / (height * width)
        cards.append(
            "<figure>"
            f"<img src='{data_uri(cut, VIEW_CROP_MAX)}'>"
            f"<figcaption><b><span class='swatch' style='background:{swatch}'></span>"
            f"{i}: {html.escape(labels[i])}</b>"
            f"<span>{area:.2f}% of frame &middot; bbox {bboxes[i]}</span>"
            "</figcaption></figure>"
        )

    overlay = data_uri(draw_overlay(image, masks, labels), VIEW_OVERLAY_MAX)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>SAM 3 &mdash; image {idx + 1}</title>"
        f"<style>{VIEWER_CSS}</style></head><body>"
        + "".join(header)
        + f"<img class='overlay' src='{overlay}'>"
        + "<div class='grid'>"
        + "".join(cards)
        + "</div></body></html>"
    )
