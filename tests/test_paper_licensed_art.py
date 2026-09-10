"""Artwork integrity tests: no model/data access or new scientific analysis."""

import importlib.util
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/paper_licensed_art.py"
SPEC = importlib.util.spec_from_file_location("paper_licensed_art", SCRIPT)
ART = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ART)


@pytest.mark.parametrize("name", ["rna", "local", "cell"])
def test_licensed_art_is_vector_only_credited_and_self_contained(name):
    root = ART.artwork(name)
    text = ET.tostring(root, encoding="unicode")
    assert "CC BY" in text and "creativecommons.org" in text
    assert "DBCLS" in text and "hakushijaco" in text
    ids = [node.get("id") for node in root.iter() if node.get("id")]
    assert len(ids) == len(set(ids))
    refs = re.findall(r"url\(#([^\)]+)\)", text)
    assert set(refs).issubset(ids)
    for node in root.iter():
        assert node.tag.split("}")[-1] not in {
            "image",
            "script",
            "foreignObject",
            "text",
            "style",
        }
    if name == "cell":
        assert "Servier Medical Art" in text


def test_source_files_match_pinned_hashes():
    for name in ART.SOURCE_HASHES:
        ART.original(name)  # Loading rejects modified or untrusted originals.


def test_local_crop_highlights_the_existing_corresponding_base_shapes():
    source = ART.artwork("rna")
    local = ART.artwork("local")
    assert source.get("viewBox") != local.get("viewBox")
    assert [node.get("d") for node in source.iter(ART.tag("path"))] == [
        node.get("d") for node in local.iter(ART.tag("path"))
    ]
    top = [
        node
        for node in local.iter(ART.tag("polygon"))
        if node.get("points", "").startswith("151.25 ")
    ]
    partner = [
        node
        for node in local.iter(ART.tag("polygon"))
        if node.get("points", "").startswith("156.17 ")
    ]
    assert len(top) == len(partner) == 1
    assert top[0].get("fill") == "#2A9D67"
    assert partner[0].get("stroke") == "#2A9D67"


def test_nested_sources_do_not_collide_when_reused(tmp_path):
    fig, ax = plt.subplots(figsize=(6, 3))
    ART.place_artwork(ax, "cell", (0, 0, 0.45, 1))
    ART.place_artwork(ax, "cell", (0.5, 0, 0.45, 1))
    path = tmp_path / "figure.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    ART.embed_licensed_art(path)
    root = ET.parse(path).getroot()
    ids = [node.get("id") for node in root.iter() if node.get("id")]
    assert len(ids) == len(set(ids))
    assert not list(root.iter(ART.tag("image")))
    for group in root.iter(ART.tag("g")):
        if group.get("id", "").startswith("licensed_art__"):
            assert group.find(ART.tag("svg")) is not None
            assert group.find(ART.tag("path")) is None


def test_unknown_asset_is_rejected():
    with pytest.raises(ValueError, match="Unknown licensed artwork"):
        ART.artwork("unreviewed_source")
