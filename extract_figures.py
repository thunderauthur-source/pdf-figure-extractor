# -*- coding: utf-8 -*-
"""Extract figure regions from local PDFs by locating figure captions.

The script reads the PDF text layer, finds caption blocks such as
"Figure 3." or "Fig. 1.", determines the column and the side of the
caption, unions the nearby raster and vector graphics, and renders the
result at a fixed resolution.

It is a reference implementation of the workflow described in README.md.
It does not download PDFs, does not use OCR, and does not split panels.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image

CAPTION_RE = re.compile(
    r"^\s*(?:fig(?:ure)?\.?)\s*(\d+)\s*(?:[.:|]|\s*[\(（])",
    re.IGNORECASE,
)
TABLE_RE = re.compile(r"^\s*table\s+[0-9ivx]+\b", re.IGNORECASE)

MARGIN_X = 18.0
MARGIN_TOP = 40.0
MARGIN_BOTTOM = 34.0


@dataclass
class Figure:
    source_pdf: str
    page: int
    number: int
    caption: str
    side: str
    column: str
    rect: fitz.Rect
    file: str = ""
    width_px: int = 0
    height_px: int = 0


def safe_name(value: str, limit: int = 80) -> str:
    """Replace characters that are invalid in Windows file names."""
    value = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", value)
    value = re.sub(r"\s+", "_", value.strip())
    return value[:limit].strip("._") or "untitled"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_gutter(page: fitz.Page) -> float:
    """Return the x coordinate of the main column gutter.

    Text blocks are rasterized into a 1D coverage array. The widest
    empty run in the central part of the page is taken as the gutter.
    Pages without a clear gutter are treated as single column.
    """
    width = page.rect.width
    spans: list[tuple[float, float]] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1 = block[:4]
        text = block[4] if len(block) > 4 else ""
        if not text.strip():
            continue
        if y1 < MARGIN_TOP or y0 > page.rect.height - MARGIN_BOTTOM:
            continue
        if x1 - x0 > 0.9 * width:
            continue
        spans.append((x0, x1))
    if not spans:
        return width / 2

    step = 2.0
    count = int(width / step) + 1
    covered = bytearray(count)
    for x0, x1 in spans:
        start = max(0, int(x0 / step))
        end = min(count, int(x1 / step) + 1)
        for index in range(start, end):
            covered[index] = 1

    low = int(0.30 * width / step)
    high = int(0.70 * width / step)
    best_len = 0
    best_center = None
    index = low
    while index < high:
        if covered[index]:
            index += 1
            continue
        end = index
        while end < high and not covered[end]:
            end += 1
        if end - index > best_len:
            best_len = end - index
            best_center = (index + end) / 2 * step
        index = end

    if best_center is None or best_len * step < 10:
        return width / 2
    return best_center


def page_graphics(page: fitz.Page, tables: list[fitz.Rect]) -> list[fitz.Rect]:
    """Collect raster and vector graphic rectangles, with page furniture removed."""
    width = page.rect.width
    height = page.rect.height
    rects: list[fitz.Rect] = []

    for info in page.get_image_info():
        rects.append(fitz.Rect(info["bbox"]))

    try:
        rects.extend(fitz.Rect(rect) for rect in page.cluster_drawings())
    except Exception:
        for drawing in page.get_drawings():
            rects.append(fitz.Rect(drawing["rect"]))

    kept: list[fitz.Rect] = []
    for rect in rects:
        if rect.is_empty or rect.is_infinite:
            continue
        w, h = rect.width, rect.height
        if w < 6 or h < 6:
            continue
        if rect.y1 < MARGIN_TOP or rect.y0 > height - MARGIN_BOTTOM:
            continue
        if w > 0.95 * width and h > 0.95 * height:
            continue
        if w > 0.5 * width and h < 15:
            continue
        if h > 0.5 * height and w < 15:
            continue
        if any(rect.intersects(table)
               and (rect & table).width * (rect & table).height > 0.5 * rect.width * rect.height
               for table in tables):
            continue
        kept.append(rect)
    return kept


def page_tables(page: fitz.Page, captions: list[tuple[fitz.Rect, int, str]]) -> list[fitz.Rect]:
    """Return table regions from captions, table backgrounds and ruled tables.

    A figure caption is never allowed to sit inside a table region: when the
    table detector returns a box that swallows the figure, that box is dropped.
    """
    width = page.rect.width
    filled: list[fitz.Rect] = []
    for drawing in page.get_drawings():
        if drawing.get("fill") is None:
            continue
        rect = fitz.Rect(drawing["rect"])
        if not (rect.width >= 100 and rect.height >= 20 and rect.width < 0.98 * width):
            continue
        if any(
            (rect & caption[0]).width * (rect & caption[0]).height
            > 0.5 * caption[0].width * caption[0].height
            for caption in captions
        ):
            continue
        filled.append(rect)

    bands: list[fitz.Rect] = []
    for block in page.get_text("blocks"):
        rect = fitz.Rect(block[:4])
        text = normalize(block[4] if len(block) > 4 else "")
        if not TABLE_RE.match(text):
            continue
        band = fitz.Rect(rect)
        for drawing in filled:
            if drawing.contains(rect):
                band |= drawing
        bands.append(band)

    try:
        for table in page.find_tables().tables:
            box = fitz.Rect(table.bbox)
            if box.width * box.height > 0.5 * page.rect.width * page.rect.height:
                continue
            if any(
                (box & caption[0]).width * (box & caption[0]).height
                > 0.5 * caption[0].width * caption[0].height
                for caption in captions
            ):
                continue
            bands.append(box)
    except Exception:
        pass
    return bands


def caption_blocks(page: fitz.Page) -> list[tuple[fitz.Rect, int, str]]:
    """Return caption rectangles, figure numbers and caption text."""
    found: list[tuple[fitz.Rect, int, str]] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1 = block[:4]
        text = normalize(block[4] if len(block) > 4 else "")
        match = CAPTION_RE.match(text)
        if not match:
            continue
        if len(text) < 30:
            continue
        found.append((fitz.Rect(x0, y0, x1, y1), int(match.group(1)), text))

    # Some PDFs repeat a caption in overlapping text layers.
    unique: list[tuple[fitz.Rect, int, str]] = []
    for item in found:
        if any(item[1] == other[1] and abs(item[0].y0 - other[0].y0) < 3 for other in unique):
            continue
        unique.append(item)
    return unique


def column_bounds(rect: fitz.Rect, page: fitz.Page, gutter: float) -> tuple[str, float, float]:
    width = page.rect.width
    if rect.width > 0.62 * width:
        return "full", MARGIN_X, width - MARGIN_X
    center = (rect.x0 + rect.x1) / 2
    if center < gutter:
        return "left", MARGIN_X, max(MARGIN_X + 40, gutter - 3)
    return "right", min(width - MARGIN_X - 40, gutter + 3), width - MARGIN_X


def overlap_x(a: fitz.Rect, b: fitz.Rect) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def vertical_gap(a: fitz.Rect, b: fitz.Rect) -> float:
    if a.y1 < b.y0:
        return b.y0 - a.y1
    if b.y1 < a.y0:
        return a.y0 - b.y1
    return 0.0


def union(rects: list[fitz.Rect]) -> fitz.Rect:
    result = fitz.Rect(rects[0])
    for rect in rects[1:]:
        result |= rect
    return result


def split_at_caption(
    rect: fitz.Rect, caption: fitz.Rect, top: float, bottom: float
) -> tuple[list[fitz.Rect], list[fitz.Rect]]:
    """Split a graphic rect that overlaps the caption into above and below parts."""
    above: list[fitz.Rect] = []
    below: list[fitz.Rect] = []
    if rect.y0 < caption.y0 - 4:
        part = fitz.Rect(
            rect.x0, max(rect.y0, top), rect.x1, min(rect.y1, caption.y0)
        )
        if part.height >= 20:
            above.append(part)
    if rect.y1 > caption.y1 + 4:
        part = fitz.Rect(
            rect.x0, max(rect.y0, caption.y1), rect.x1, min(rect.y1, bottom)
        )
        if part.height >= 20:
            below.append(part)
    return above, below


def pick_region(
    graphics: list[fitz.Rect],
    caption: fitz.Rect,
    x0: float,
    x1: float,
    top: float,
    bottom: float,
    max_gap: float,
    page_width: float,
) -> tuple[fitz.Rect | None, str]:
    """Choose the graphic group belonging to one caption."""
    def in_column(rect: fitz.Rect) -> bool:
        # Figures that span both columns belong to captions in either column.
        if rect.width > 0.5 * page_width:
            return rect.x0 < x1 and rect.x1 > x0
        center = (rect.x0 + rect.x1) / 2
        return x0 <= center <= x1

    candidates = [rect for rect in graphics if in_column(rect)]
    # A tall, narrow caption is the signature of a side-caption layout
    # (Science and several Nature journals put the caption beside the figure).
    side_caption = caption.width <= 0.35 * page_width and caption.height >= 80

    above: list[fitz.Rect] = []
    below: list[fitz.Rect] = []
    side_left: list[fitz.Rect] = []
    side_right: list[fitz.Rect] = []
    clip_left: list[fitz.Rect] = []
    clip_right: list[fitz.Rect] = []

    for rect in candidates:
        overlaps_caption = min(rect.y1, caption.y1) - max(rect.y0, caption.y0)
        if overlaps_caption > 0:
            parts_above, parts_below = split_at_caption(rect, caption, top, bottom)
            above.extend(parts_above)
            below.extend(parts_below)
            vertical_overlap = overlaps_caption
            if vertical_overlap >= 0.5 * caption.height:
                if rect.x0 >= caption.x1 - 2:
                    side_right.append(rect)
                elif rect.x1 <= caption.x0 + 2:
                    side_left.append(rect)
                elif side_caption:
                    # The graphic cluster may wrap the caption as well; keep
                    # only the part beside the caption. These clipped boxes
                    # are used only when no clean side box exists.
                    right = fitz.Rect(
                        max(rect.x0, caption.x1 + 2), rect.y0, rect.x1, rect.y1
                    )
                    if right.width >= 100:
                        clip_right.append(right)
                    left = fitz.Rect(
                        rect.x0, rect.y0, min(rect.x1, caption.x0 - 2), rect.y1
                    )
                    if left.width >= 100:
                        clip_left.append(left)
            continue

        if rect.y1 <= caption.y0 + 2 and rect.y0 >= top:
            if caption.y0 - rect.y1 <= max_gap:
                above.append(rect)
        elif rect.y0 >= caption.y1 - 2 and rect.y1 <= bottom:
            if rect.y0 - caption.y1 <= max_gap:
                below.append(rect)

    def group_area(rects: list[fitz.Rect]) -> float:
        if not rects:
            return 0.0
        box = union(rects)
        return box.width * box.height

    if clip_right and (
        not side_right or group_area(side_right) < 0.5 * group_area(clip_right)
    ):
        side_right = clip_right
    if clip_left and (
        not side_left or group_area(side_left) < 0.5 * group_area(clip_left)
    ):
        side_left = clip_left

    area_above = group_area(above)
    area_below = group_area(below)
    area_left = group_area(side_left)
    area_right = group_area(side_right)

    side = "none"
    if area_left or area_right:
        if area_left >= area_right:
            side = "left"
        else:
            side = "right"
    if side != "none" and (area_left + area_right) >= 0.5 * max(area_above, area_below, 1):
        region = union(side_left if side == "left" else side_right)
        region.y0 = max(region.y0, top)
        region.y1 = min(region.y1, bottom)
        region.x0 = max(region.x0, MARGIN_X)
        region.x1 = min(region.x1, page_width - MARGIN_X)
        return region, side

    if not above and not below:
        return None, "none"

    side = "above" if area_above >= area_below else "below"
    chosen = above if side == "above" else below

    def distance(rect: fitz.Rect) -> float:
        return caption.y0 - rect.y1 if side == "above" else rect.y0 - caption.y1

    chosen.sort(key=lambda rect: (max(0.0, distance(rect)), -rect.width * rect.height))
    anchor = chosen[0]

    group = [anchor]
    changed = True
    while changed:
        changed = False
        for rect in chosen:
            if rect in group:
                continue
            if any(
                overlap_x(rect, member) > 0.3 * min(rect.width, member.width)
                and vertical_gap(rect, member) < 35
                for member in group
            ):
                group.append(rect)
                changed = True

    region = union(group)
    if side == "above":
        region.y1 = min(region.y1, caption.y0 - 1)
    else:
        region.y0 = max(region.y0, caption.y1 + 1)
    # A figure that spans the gutter keeps its full width.
    if region.width > 0.5 * page_width:
        region.x0 = max(region.x0, MARGIN_X)
        region.x1 = min(region.x1, page_width - MARGIN_X)
    else:
        region.x0 = max(region.x0, x0)
        region.x1 = min(region.x1, x1)
    region.y0 = max(region.y0, top, MARGIN_TOP)
    region.y1 = min(region.y1, caption.y0 if side == "above" else bottom)
    return region, side


def extract_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    min_area: float,
    quality: int,
    max_gap: float,
) -> list[Figure]:
    doc = fitz.open(pdf_path)
    if doc.needs_pass:
        print(f"skip encrypted: {pdf_path}", file=sys.stderr)
        return []

    source = safe_name(pdf_path.stem, 60)
    figures: list[Figure] = []
    zoom = dpi / 72.0

    for page_index, page in enumerate(doc, start=1):
        captions = caption_blocks(page)
        if not captions:
            continue
        captions.sort(key=lambda item: item[0].y0)
        tables = page_tables(page, captions)
        graphics = page_graphics(page, tables)
        if not graphics:
            continue
        gutter = find_gutter(page)
        page_area = page.rect.width * page.rect.height

        for index, (caption_rect, number, caption) in enumerate(captions):
            column, x0, x1 = column_bounds(caption_rect, page, gutter)

            same_column = [
                item for item in captions
                if column_bounds(item[0], page, gutter)[0] == column
            ]
            position = same_column.index((caption_rect, number, caption))
            top = MARGIN_TOP
            bottom = page.rect.height - MARGIN_BOTTOM
            if position > 0:
                top = max(top, same_column[position - 1][0].y1 + 2)
            if position + 1 < len(same_column):
                bottom = min(bottom, same_column[position + 1][0].y0 - 2)
            for table in tables:
                if table.x1 < x0 or table.x0 > x1:
                    continue
                if table.y1 <= caption_rect.y0:
                    top = max(top, table.y1 + 4)
                elif table.y0 >= caption_rect.y1:
                    bottom = min(bottom, table.y0 - 4)

            region, side = pick_region(
                graphics, caption_rect, x0, x1, top, bottom, max_gap, page.rect.width
            )
            if region is None or region.is_empty:
                continue
            region = fitz.Rect(
                max(0, region.x0 - 3),
                max(0, region.y0 - 3),
                min(page.rect.width, region.x1 + 3),
                min(page.rect.height, region.y1 + 3),
            )
            # The margin expansion must not reach back into a side caption.
            for cap_rect, _, _ in captions:
                if cap_rect.y1 <= region.y0 or cap_rect.y0 >= region.y1:
                    continue
                if cap_rect.x0 < region.x0 < cap_rect.x1:
                    region.x0 = cap_rect.x1 + 1
                elif cap_rect.x0 < region.x1 < cap_rect.x1:
                    region.x1 = cap_rect.x0 - 1
            # Do not let the margin expansion cut into a line of text.
            for block in page.get_text("blocks"):
                bx0, by0, bx1, by1 = block[:4]
                if bx1 <= region.x0 + 1 or bx0 >= region.x1 - 1:
                    continue
                if region.y0 - 6 <= by1 <= region.y0 + 6:
                    region.y0 = max(region.y0, by1 + 1)
                if region.y1 - 6 <= by0 <= region.y1 + 6:
                    region.y1 = min(region.y1, by0 - 1)
            if region.width < 40 or region.height < 40:
                continue
            if region.width * region.height / page_area < min_area:
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=region, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            file_name = f"Fig{number:02d}_{source}_p{page_index:03d}"
            if any(fig.file.startswith(file_name) for fig in figures):
                file_name += f"_{index + 1}"
            file_name += ".jpg"

            pdf_out = out_dir / source
            pdf_out.mkdir(parents=True, exist_ok=True)
            image.save(pdf_out / file_name, "JPEG", quality=quality)

            figures.append(
                Figure(
                    source_pdf=pdf_path.name,
                    page=page_index,
                    number=number,
                    caption=caption[:300],
                    side=side,
                    column=column,
                    rect=region,
                    file=str((pdf_out / file_name).relative_to(out_dir)),
                    width_px=pix.width,
                    height_px=pix.height,
                )
            )

    doc.close()
    return figures


def write_index(out_dir: Path, figures: list[Figure]) -> None:
    path = out_dir / "_index.csv"
    fields = [
        "source_pdf", "page", "figure", "side", "column",
        "x0_pt", "y0_pt", "x1_pt", "y1_pt",
        "width_px", "height_px", "file", "caption",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fig in figures:
            writer.writerow({
                "source_pdf": fig.source_pdf,
                "page": fig.page,
                "figure": fig.number,
                "side": fig.side,
                "column": fig.column,
                "x0_pt": round(fig.rect.x0, 1),
                "y0_pt": round(fig.rect.y0, 1),
                "x1_pt": round(fig.rect.x1, 1),
                "y1_pt": round(fig.rect.y1, 1),
                "width_px": fig.width_px,
                "height_px": fig.height_px,
                "file": fig.file,
                "caption": fig.caption,
            })


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract figure regions from local PDFs by caption position."
    )
    parser.add_argument("pdf_dir", type=Path, help="directory containing PDF files")
    parser.add_argument("-o", "--out", type=Path, default=Path("out"), help="output directory")
    parser.add_argument("--dpi", type=int, default=320, help="render resolution (default 320)")
    parser.add_argument("--min-area", type=float, default=0.03,
                        help="minimum figure area as a fraction of the page (default 0.03)")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality (default 92)")
    parser.add_argument("--max-gap", type=float, default=120,
                        help="maximum caption-to-figure gap in points (default 120)")
    args = parser.parse_args()

    pdf_dir = args.pdf_dir.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    if not pdf_dir.is_dir():
        parser.error(f"not a directory: {pdf_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        parser.error(f"no PDF files under: {pdf_dir}")

    all_figures: list[Figure] = []
    for number, pdf_path in enumerate(pdfs, start=1):
        print(f"[{number}/{len(pdfs)}] {pdf_path.name}", flush=True)
        figures = extract_pdf(
            pdf_path, out_dir, args.dpi, args.min_area, args.quality, args.max_gap
        )
        print(f"    figures: {len(figures)}", flush=True)
        all_figures.extend(figures)

    write_index(out_dir, all_figures)
    print(f"total figures: {len(all_figures)}")
    print(f"index: {out_dir / '_index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
