#!/usr/bin/env python3
# ruff: noqa: ISC004
"""Generate non-quantitative figure components with Azure GPT Image 1.5.

The script intentionally creates isolated, transparent illustration assets. All
quantitative marks, labels, legends, and arrows are added later by the vector
figure renderer. Credentials are read from AZURE_OPENAI_API_KEY and are never
written to disk.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

DEFAULT_ENDPOINT = (
    "https://galor-m6d0ej1n-eastus2.cognitiveservices.azure.com/openai/"
    "deployments/gpt-image-1.5/images/generations?api-version=2024-02-01"
)

COMMON = (
    " Isolated publication-quality scientific illustration asset on a native "
    "transparent background. Refined Nature/PNAS biology-figure aesthetic, "
    "precise scientific visual language, thin charcoal outlines, restrained "
    "navy teal amber coral palette, subtle dimensionality, no photorealism, "
    "no text, no letters, no numerals, no labels, no arrows, no border, no "
    "chart, no decorative background, no watermark, generous transparent margin."
)

FAMILIES = {
    "rna_duplex": [
        "A compact endogenous double-stranded RNA editing substrate: two softly "
        "twisting antiparallel RNA strands with recognizable paired nucleotides; "
        "one target nucleotide emphasized by a small warm amber halo and a subtle "
        "base-conversion cue. Do not depict a protein or chemical atom model.",
        "A graceful imperfect RNA duplex formed by two folded transcript arms, "
        "with short paired rungs, two small bulges, and one highlighted target site; "
        "clean editorial cutaway, biologically plausible but deliberately schematic.",
        "An elegant horizontal RNA hairpin/duplex motif, paired rails and base-pair "
        "rungs, one focal nucleotide glowing amber, a few mismatches visible as "
        "small gaps; compact silhouette suitable beside vector annotations.",
    ],
    "cell_context": [
        "A cutaway mammalian cell and nucleus containing several nascent RNA strands, "
        "a highlighted nuclear double-stranded RNA substrate, and a few abstract "
        "regulatory protein silhouettes nearby. Scientifically restrained; proteins "
        "must be generic shapes, not claimed atomic structures.",
        "A compact cellular-context vignette: translucent cell membrane and nucleus, "
        "chromatin threads, one emerging transcript folding into double-stranded RNA, "
        "and small generic regulator particles with varied accessibility.",
        "A polished nuclear RNA-processing scene viewed in cross-section, showing a "
        "single RNA duplex embedded among transcript and chromatin context, with "
        "subtle spatial gradients that suggest cell state without quantitative encoding.",
    ],
    "human_contexts": [
        "A coordinated set of four small scientific tissue glyphs arranged in one row: "
        "arterial vessel cross-section, human brain silhouette, liver lobe, and skeletal "
        "muscle fibers. Equal visual weight, anatomically recognizable, icon-like, no text.",
        "Four refined biomedical mini-illustrations in a horizontal constellation: blood "
        "vessel wall, cerebral cortex/brain, liver, and striated skeletal muscle. Consistent "
        "scale and style, separated by transparent space, no connecting arrows.",
        "A compact atlas strip of artery, brain, liver, and skeletal muscle, each rendered "
        "as a clean scientific cutaway icon with a shared palette and no labels; accurate "
        "organ silhouettes, suitable for a genomics figure margin.",
    ],
    "marine_lineages": [
        "A coordinated scientific silhouette trio arranged horizontally: octopus, acorn "
        "worm of genus Ptychodera with elongated proboscis-collar-trunk body plan, and sea "
        "urchin of genus Strongylocentrotus. Accurate recognizable forms, equal visual weight.",
        "Three restrained marine-animal natural-history icons: an octopus with eight arms, "
        "a slender acorn worm showing proboscis and collar, and a round spiny sea urchin. "
        "Editorial biology-figure style, horizontally separated on transparency.",
        "A compact phylogenetic-context vignette with isolated silhouettes of octopus, "
        "Ptychodera acorn worm, and Strongylocentrotus sea urchin; scientifically faithful "
        "body shapes, unified linework, no tree, no labels.",
    ],
    "latent_context": [
        "A layered scientific vignette expressing unmeasured cellular state: a crisp RNA "
        "duplex in front, partially visible nucleus and transcript isoforms behind it, and "
        "several softly fading regulator silhouettes at the periphery. The fading is purely "
        "conceptual and must not imply measured quantities.",
        "A conceptual missing-information asset for RNA editing: a clear focal RNA substrate "
        "surrounded by translucent layers of cell state, alternative transcript paths, "
        "folded conformations, and generic regulatory particles, some fading into transparency.",
        "An elegant biological uncertainty landscape centered on a target RNA duplex: nearby "
        "cellular and conformational influences are crisp, while more distant donor-state and "
        "regulator-context motifs dissolve softly; no question marks or symbolic text.",
    ],
}

SELECTIONS = {
    "rna_duplex": {
        "candidate": 3,
        "rationale": "Most legible horizontal duplex and focal site after reduction.",
    },
    "cell_context": {
        "candidate": 2,
        "rationale": "Cleanest nuclear context with the least extraneous anatomy.",
    },
    "human_contexts": {
        "candidate": 1,
        "rationale": "Most coherent and recognizable four-tissue atlas strip.",
    },
    "marine_lineages": {
        "candidate": 3,
        "rationale": "All three organisms are complete and anatomically distinguishable.",
    },
    "latent_context": {
        "candidate": 1,
        "rationale": "Layering conveys unavailable context without quantitative symbolism.",
    },
}


def generate_one(
    endpoint: str,
    api_key: str,
    family: str,
    index: int,
    prompt: str,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, object]:
    output_path = output_dir / f"{family}_candidate_{index:02d}.png"
    full_prompt = prompt + COMMON
    record: dict[str, object] = {
        "family": family,
        "candidate": index,
        "path": output_path.as_posix(),
        "prompt": full_prompt,
        "prompt_sha256": hashlib.sha256(full_prompt.encode()).hexdigest(),
        "model_deployment": "gpt-image-1.5",
        "size": "1024x1024",
        "quality": "high",
        "background": "transparent",
    }
    if output_path.exists() and not overwrite:
        record["status"] = "existing"
        record["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
        return record

    body = json.dumps(
        {
            "prompt": full_prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "high",
            "output_format": "png",
            "background": "transparent",
        }
    ).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"{family} candidate {index}: HTTP {error.code}: {details}")
    image_data = payload.get("data", [{}])[0].get("b64_json")
    if not image_data:
        raise RuntimeError(f"{family} candidate {index}: response omitted b64_json")
    output_path.write_bytes(base64.b64decode(image_data))
    record["status"] = "generated"
    record["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return record


def curate_assets(output_dir: Path) -> None:
    """Crop transparent margins and retain the prereviewed candidate per family."""
    selected_dir = output_dir.parent / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selections = []
    for family, specification in SELECTIONS.items():
        index = int(specification["candidate"])
        source = output_dir / f"{family}_candidate_{index:02d}.png"
        target = selected_dir / f"{family}.png"
        with Image.open(source) as image:
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            bounds = alpha.getbbox()
            if bounds is None:
                raise RuntimeError(f"Selected asset has an empty alpha channel: {source}")
            rgba.crop(bounds).save(target)
        selections.append(
            {
                "family": family,
                "selected_candidate": index,
                "source": source.as_posix(),
                "asset": target.as_posix(),
                "rationale": specification["rationale"],
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    selection_path = output_dir.parent / "selection_manifest.json"
    selection_path.write_text(
        json.dumps(
            {
                "review_standard": (
                    "PNAS/Nature-style legibility, biological restraint, clean alpha "
                    "boundary, and utility beneath vector annotation"
                ),
                "selections": selections,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {selection_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figure_assets/candidates"),
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AZURE_OPENAI_IMAGE_ENDPOINT", DEFAULT_ENDPOINT),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("AZURE_OPENAI_API_KEY is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (family, index, prompt)
        for family, prompts in FAMILIES.items()
        for index, prompt in enumerate(prompts, start=1)
    ]
    records: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                generate_one,
                args.endpoint,
                api_key,
                family,
                index,
                prompt,
                args.output_dir,
                args.overwrite,
            ): (family, index)
            for family, index, prompt in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            family, index = futures[future]
            record = future.result()
            records.append(record)
            print(f"{record['status']}: {family} candidate {index}", flush=True)

    records.sort(key=lambda item: (str(item["family"]), int(item["candidate"])))
    manifest = {
        "purpose": (
            "Non-quantitative illustrative components; all data marks and annotations "
            "are composed separately in vector form."
        ),
        "assets": records,
    }
    manifest_path = args.output_dir.parent / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    curate_assets(args.output_dir)


if __name__ == "__main__":
    main()
