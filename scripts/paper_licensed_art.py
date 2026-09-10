"""Adapt credited scientific SVGs without rasterization or image generation.

Originals, licenses, and semantic limits: paper/illustration_sources/README.md.
The generic RNA motif is not an observed Alu or a sequence-specific structure.
"""

from __future__ import annotations

import copy
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from matplotlib.patches import Rectangle

SVG = "http://www.w3.org/2000/svg"
ORIGINALS = Path(__file__).resolve().parents[1] / "paper/illustration_sources/originals"
SOURCE_HASHES = {
    "dbcls_pri_mirna.svg": "77e45cc25440997f2c6279bf2d4be35c24b2cf5ae3a511052361903bad295da8",
    "servier_cell_membrane.svg": "0d746b099fe669cde2dcc9dbb4e89ff6da6b7e86d5347847f477c0aefb782bfb",
    "servier_nucleus.svg": "534948dd77f32890efce6924dc3ae15a707c41cc56696cb48b505d9847e1a4d4",
}
RNA_CREDIT = (
    "Generic RNA motif adapted from pri-miRNA, designed by @hakushijaco, "
    "edited by Hiromasa Ono, DBCLS/TogoTV, https://doi.org/10.7875/togopic.2024.065, "
    "CC BY 4.0 https://creativecommons.org/licenses/by/4.0/. "
    "Cropped, recolored, and annotated; no specific RNA sequence or conformation."
)
CELL_CREDIT = (
    "Cell membrane and nucleus adapted from Servier Medical Art "
    "https://smart.servier.com/, via Bioicons revision "
    "d29e766ea7580b8063c4f47b29e872db40a4d979, CC BY 3.0 "
    "https://creativecommons.org/licenses/by/3.0/. Recolored and recomposed. "
    "Conceptual compartments, not measured localization. " + RNA_CREDIT
)
NUMBER = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def tag(name):
    return f"{{{SVG}}}{name}"


def original(name):
    data = (ORIGINALS / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[name]:
        raise ValueError(f"Licensed source changed: {name}")
    root = ET.fromstring(data)
    # Inline the source's simple class rules, avoiding CSS collisions between
    # multiple nested illustrations or with Matplotlib's figure-wide styles.
    classes = {}
    for style in root.iter(tag("style")):
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", style.text or ""):
            declarations = {
                key.strip(): value.strip()
                for key, value in re.findall(r"([\w-]+)\s*:\s*([^;]+)", body)
            }
            for selector in selectors.split(","):
                classes.setdefault(selector.strip().lstrip("."), {}).update(
                    declarations
                )
    for node in root.iter():
        if node.tag.split("}")[-1] in {"image", "script", "foreignObject", "text"}:
            raise ValueError(f"Unexpected non-illustration content in {name}")
        if any("href" in key for key in node.attrib):
            raise ValueError(f"External reference in {name}")
        for cls in node.attrib.pop("class", "").split():
            for key, value in classes.get(cls, {}).items():
                node.set(key, value)
        node.attrib.pop("data-name", None)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == tag("style"):
                parent.remove(child)
    root.attrib.pop("id", None)
    ids = {node.get("id") for node in root.iter() if node.get("id")}
    prefix = Path(name).stem + "_"
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if key == "id":
                node.set(key, prefix + value)
            else:
                for ident in ids:
                    value = value.replace(f"url(#{ident})", f"url(#{prefix}{ident})")
                node.set(key, value)
    return root


def recolor(root, palette):
    for node in root.iter():
        for key in ("fill", "stroke"):
            color = node.get(key)
            if color in palette:
                node.set(key, palette[color])


def rna(local=False):
    root = original("dbcls_pri_mirna.svg")
    recolor(
        root,
        {
            "#6278e6": "#65899E",
            "#87c0ff": "#A9C6D4",
            "#89d0fc": "#C2D8E1",
            "#68cdf2": "#426F8D",
            "#6898f2": "#86ADBF",
        },
    )
    root.set("viewBox", "96 168 112 64" if local else "0 150 400 96")
    root.set("overflow", "hidden")
    ET.SubElement(root, tag("desc")).text = RNA_CREDIT
    if local:
        # Corresponding source base shapes, not a new pairing or sequence claim.
        for node in root.iter(tag("polygon")):
            points = node.get("points", "")
            if points.startswith("151.25 "):
                node.set("fill", "#2A9D67")
            elif points.startswith("156.17 "):
                node.set("fill", "#F3FAF6")
                node.set("stroke", "#2A9D67")
                node.set("stroke-width", "0.7")
    return root


def cell():
    root = ET.Element(tag("svg"), {"viewBox": "0 0 305 282"})
    ET.SubElement(root, tag("desc")).text = CELL_CREDIT
    membrane = original("servier_cell_membrane.svg")
    palette = {
        "#7dcc5a": "#B9CCD5",
        "#63be3a": "#91ABB9",
        "#9fd985": "#CFDDE4",
        "#fffa8e": "#E1EAF0",
        "#f9f390": "#CBDCE5",
        "#eaecf4": "#EBF1F5",
        "#333": "#708692",
    }
    # Preserve the illustrator's lightness gradients while reducing saturation.
    for node in membrane.iter():
        color = node.get("fill", "")
        if re.fullmatch(r"#[a-f0-9]{6}", color) and color.startswith(
            ("#fa", "#fb", "#fc", "#fd")
        ):
            amount = min(1, max(0, (int(color[-2:], 16) - 168) / 49))
            rgb = [
                round(a + amount * (b - a))
                for a, b in zip((235, 242, 246), (250, 252, 253))
            ]
            palette[color] = "#" + "".join(f"{value:02x}" for value in rgb)
    recolor(membrane, palette)
    membrane.set("x", "2")
    membrane.set("y", "4")
    membrane.set("width", "294")
    membrane.set("height", "272")
    root.append(membrane)

    nucleus = original("servier_nucleus.svg")
    recolor(
        nucleus,
        {
            "#cf0267": "#D8DCE8",
            "#f6dde8": "#F2F1F8",
            "#b9015c": "#BEC5D6",
            "#c83e80": "#939FBC",
            "#ef92be": "#D9DDEC",
            "#d15a92": "#A4AFC8",
            "#feeff6": "#FAFAFD",
            "#e16ca4": "#B2BBD2",
            "#ee82b4": "#D4DBE8",
            "#333": "#748099",
        },
    )
    nucleus.set("x", "66")
    nucleus.set("y", "66")
    nucleus.set("width", "129")
    nucleus.set("height", "143")
    root.append(nucleus)
    # Two RNA motifs identify the conceptual compartments. No connecting line
    # or arrow implies transport, editing, or a measured localization pattern.
    inner = rna()
    inner.attrib.update(x="86", y="120", width="84", height="29")
    root.append(inner)
    cytoplasmic = rna()
    cytoplasmic.attrib.update(x="155", y="207", width="85", height="26")
    root.append(cytoplasmic)
    return root


def artwork(name):
    if name == "rna":
        return rna()
    if name == "local":
        return rna(local=True)
    if name == "cell":
        return cell()
    raise ValueError(f"Unknown licensed artwork: {name}")


def place_artwork(ax, name, bounds, *, zorder=2):
    """Reserve a fitted SVG viewport; return its axes-relative bounds.

    The invisible vector rectangle is replaced after Matplotlib serializes its
    exact page position. No raster proxy or traced image enters the build.
    """
    x, y, width, height = bounds
    ratios = {"rna": 400 / 96, "local": 112 / 64, "cell": 305 / 282}
    box = ax.get_position()
    ax_w = ax.figure.get_figwidth() * box.width
    ax_h = ax.figure.get_figheight() * box.height
    target = ratios[name] * ax_h / ax_w
    if width / height > target:
        fitted = height * target
        x += (width - fitted) / 2
        width = fitted
    else:
        fitted = width / target
        y += (height - fitted) / 2
        height = fitted
    placeholder = Rectangle(
        (x, y),
        width,
        height,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="none",
        zorder=zorder,
    )
    serial = getattr(ax.figure, "_licensed_art_count", 0)
    ax.figure._licensed_art_count = serial + 1
    placeholder.set_gid(f"licensed_art__{name}__{serial}")
    ax.add_patch(placeholder)
    return x, y, width, height


def embed_licensed_art(svg_path):
    tree = ET.parse(svg_path)
    for group in tree.iter(tag("g")):
        ident = group.get("id", "")
        if not ident.startswith("licensed_art__"):
            continue
        path = group.find(tag("path"))
        if path is None:
            raise ValueError(f"Missing illustration viewport {ident}")
        coords = [float(value) for value in re.findall(NUMBER, path.get("d", ""))]
        xs, ys = coords[::2], coords[1::2]
        left, top = min(xs), min(ys)
        width, height = max(xs) - left, max(ys) - top
        source = copy.deepcopy(artwork(ident.split("__")[1]))
        source.attrib.update(
            x=str(left),
            y=str(top),
            width=str(width),
            height=str(height),
            preserveAspectRatio="xMidYMid meet",
        )
        # Prefix IDs and references so repeated source clip paths stay local.
        ids = {node.get("id") for node in source.iter() if node.get("id")}
        for node in source.iter():
            for key, value in list(node.attrib.items()):
                if key == "id":
                    node.set(key, ident + "_" + value)
                else:
                    for old in ids:
                        value = value.replace(f"url(#{old})", f"url(#{ident}_{old})")
                    node.set(key, value)
        group.remove(path)
        group.append(source)
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)
