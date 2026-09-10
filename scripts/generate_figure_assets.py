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
    "transparent background. Refined Nature/PNAS molecular-biology figure aesthetic, "
    "precise scientific visual language, fine charcoal outlines, restrained "
    "navy teal amber coral palette, subtle dimensionality, no photorealism, no glossy "
    "app-icon finish, no thick cartoon outline, "
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
    "cell_context_rna": [
        "A compact cutaway mammalian cell centered on cellular context for RNA editing. "
        "Inside the cell, show one continuous orange single-stranded RNA squiggle with a "
        "small local U-shaped fold-back region indicated by only three short dotted pairing "
        "bridges and one amber target dot. Add two restrained generic RNA-binding protein "
        "shapes nearby. Do not draw a twisted helix, ladder, DNA, chromosomes, chromatin, "
        "nucleotides as colored bars, or transcription machinery.",
        "A polished cell-state vignette for A-to-I RNA editing: a translucent mammalian "
        "cell containing one clearly single-contour RNA transcript, locally folded into an "
        "asymmetric hairpin with a bulge and three subtle dashed pairing marks. Sparse "
        "generic protein shapes surround but do not cover the transcript. The RNA must look "
        "like a flexible strand, never a double helix or regular ladder. No DNA, chromosome, "
        "chromatin, replication, or transcription-fork imagery.",
        "A restrained scientific cellular-context icon with a simple cell boundary, soft "
        "cytoplasmic and nuclear regions, generic regulator particles, and a single orange "
        "RNA strand crossing the scene. Give the strand one short imperfect fold-back loop "
        "with a small bulge and a highlighted target dot, but no visible ladder or helix. "
        "Absolutely no DNA, chromosomes, chromatin threads, genomic symbols, or base bars.",
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
    "rna_substrate_editorial": [
        "A horizontal endogenous double-stranded RNA substrate formed when two regions "
        "of one flexible RNA transcript fold back and pair. Show an imperfect, nearly "
        "planar antiparallel duplex with fine ribose-phosphate rails, short base-pair "
        "bridges, one asymmetric internal bulge, and one small loop. One central adenosine "
        "may be marked by a restrained amber accent. It must read as folded RNA, not a "
        "regular DNA double helix, chromosome, molecular ball-and-stick model, or ladder.",
        "A refined editorial cutaway of an inverted-repeat RNA duplex: one continuous "
        "single-stranded transcript curves into two paired arms, with visible imperfect "
        "base pairing, a small mismatch, and a bulged nucleotide. Use an elongated low "
        "profile suitable for a journal figure. Avoid DNA helices, equal rigid rungs, "
        "chromosomes, proteins, and cellular scenery.",
        "A compact biological illustration of a long imperfect dsRNA stem embedded in a "
        "continuous transcript, shown as two gently curved RNA backbones connected by "
        "delicate hydrogen-bond cues. Include minor asymmetry and a single focal amber "
        "nucleotide. The silhouette should be horizontal, elegant, and unmistakably RNA; "
        "never depict the canonical twisted DNA double helix.",
        "A horizontal A-form double-stranded RNA substrate rendered as a restrained molecular "
        "ribbon cutaway. Show the compact helical pitch and broad shallow groove characteristic "
        "of dsRNA, short single-stranded transcript tails, one small asymmetric bulge, and one "
        "amber target nucleotide. Use fine scientific-illustration detail without atomistic clutter. "
        "Do not use the familiar symmetric B-DNA icon, rectangular colored base bars, or a ladder.",
        "A polished molecular-biology plate of an inverted-Alu RNA fold: two flexible transcript "
        "arms meet in an imperfect A-form RNA stem with short tails, uneven pairing, a small loop, "
        "and a bulged adenosine. Three-quarter horizontal view, delicate backbone texture, crisp at "
        "thumbnail scale. Explicitly avoid DNA-like symmetry, chromosomes, nucleosomes, and textbook "
        "double-helix iconography.",
        "A refined semi-flat illustration of one endogenous RNA transcript folding into a compact "
        "imperfect duplex. Preserve a visibly continuous RNA contour, a broad A-form stem, several "
        "paired contacts, one mismatch, and one warm focal nucleotide. Horizontal editorial layout; "
        "no regular DNA helix, no colored ladder, no atom spheres, and no hand-drawn doodle style.",
    ],
    "target_pair_context": [
        "A close editorial view of a short imperfect RNA duplex centered on one paired "
        "position. The upper focal adenosine is a small amber nucleotide and the directly "
        "paired lower nucleotide is teal; both sit within a visibly paired local RNA "
        "neighborhood with one nearby bulge. Keep the two highlighted nucleotides vertically "
        "aligned near the exact center and leave open space above and below for vector callouts. "
        "No DNA helix, no protein, no cell, no labels, brackets, arrows, or glow.",
        "A precise horizontal target-centered RNA pairing motif, almost planar rather than "
        "twisted: two fine antiparallel RNA strands, seven to nine local pairing contacts, "
        "an amber focal nucleotide on the upper strand and a teal predicted partner directly "
        "below it. Add one subtle mismatch outside the central pair. Clean scientific plate, "
        "with the central pair dominant and generous transparent space for annotation.",
        "A high-end molecular-biology figure insert showing an imperfect local RNA stem. "
        "Center a single amber adenosine opposite a teal partner, connected by a delicate "
        "pairing cue; flank them with several neutral paired nucleotides and a modest bulge. "
        "Use restrained flat depth and fine linework. Avoid a generic DNA ladder, helical "
        "twist, atom model, labels, arrows, halos, or decorative effects.",
        "A horizontal close-up cutaway through an imperfect A-form RNA stem, designed as a small "
        "journal figure insert. At the exact center, show one amber ribonucleotide opposite one teal "
        "ribonucleotide with a delicate pairing interaction. Neighboring paired bases and one bulge "
        "provide local context. Keep the geometry planar enough for vector labels; no DNA helix, "
        "rectangular ladder, glow, letters, brackets, or arrows.",
        "A precise molecular editorial illustration of target-centered RNA pairing: two gently curved "
        "ribose-phosphate backbones frame a short imperfect paired region. One central upper nucleotide "
        "is amber and its directly aligned lower partner is teal; all other nucleotides are restrained "
        "blue-gray. Include open transparent space around the central pair. No DNA iconography, no "
        "protein, no cell, no labels, no radiance, and no toy-like colored blocks.",
        "A high-end PNAS-style detail view of a local dsRNA neighborhood, horizontal and nearly planar. "
        "Render a central amber adenosine paired directly to a teal partner, with five flanking pairing "
        "contacts and one small asymmetric loop. Use subtle material depth and fine linework that remains "
        "clear at print scale. Avoid DNA twist, rigid rungs, geometric tokens, text, callouts, and arrows.",
    ],
    "substrate_organization": [
        "A compact horizontal concept of substrate organization in an inverted-repeat RNA: "
        "a single transcript contains two separated repeat-derived arms that curve together "
        "into one imperfect dsRNA stem, while a restrained open arc between their origins "
        "suggests inter-arm genomic separation. Show one substrate only. No chromosome, DNA "
        "double helix, scale numbers, labels, arrows, or quantitative encoding.",
        "An editorial biological vignette of two distant regions of one pre-mRNA forming an "
        "imperfect RNA duplex. Keep the paired stem crisp in the foreground and show the long "
        "intervening transcript as a soft curved path behind it, communicating pair-level "
        "organization without implying a measured mechanism. No cell, DNA, ruler, text, or icons.",
        "A polished schematic illustration of one folded transcript whose separated inverted "
        "repeat arms meet to form an asymmetric RNA duplex. Emphasize the relation between "
        "transcript-scale spacing and the paired substrate using spatial composition alone. "
        "No arrows, brackets, labels, DNA helix, chromosomes, or numeric distance marks.",
    ],
    "matched_cell_state": [
        "A compact cutaway mammalian cell representing unresolved matched cellular state for "
        "RNA editing. Show a translucent cell and nucleus, one continuous orange RNA transcript "
        "with a small imperfect fold-back duplex, two faint alternative transcript paths, and "
        "several generic regulator shapes with varied proximity. No DNA, chromosomes, chromatin "
        "fibers, transcription forks, regular double helices, labels, or question marks.",
        "A restrained scientific cell-state vignette centered on RNA: a soft cell boundary and "
        "nuclear region contain one clearly single-contour transcript, a local imperfect dsRNA "
        "fold, sparse generic RNA-binding proteins, and subtle alternative processing branches. "
        "The RNA must remain visually dominant. Exclude DNA, chromosomes, chromatin, organelle "
        "catalogues, labels, and decorative particles.",
        "A refined PNAS-style conceptual asset for transcript and cellular context at matched "
        "scale. Inside a simple mammalian cell silhouette, show one folded RNA substrate, a few "
        "translucent regulator forms, and two softly diverging transcript isoform paths. Keep "
        "the scene sparse, biologically restrained, and unmistakably about RNA rather than DNA."
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
    "cell_context_rna": {
        "candidate": 1,
        "rationale": (
            "Clearest single-contour RNA transcript, local fold-back cue, and cell-state "
            "context without a DNA-like ladder or genomic imagery."
        ),
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
    "rna_substrate_editorial": {
        "candidate": 1,
        "rationale": (
            "The continuous fold-back contour, asymmetric bulges, and low-profile paired "
            "stem remain legible after reduction without resembling canonical B-DNA."
        ),
    },
    "target_pair_context": {
        "candidate": 4,
        "rationale": (
            "The central amber/teal pair is unambiguous, the neighboring imperfect stem "
            "provides local context, and open margins support controlled vector callouts."
        ),
    },
    "substrate_organization": {
        "candidate": 1,
        "rationale": (
            "A single continuous transcript forms a compact imperfect stem while the broad "
            "arc communicates transcript-scale organization without a numeric claim."
        ),
    },
    "matched_cell_state": {
        "candidate": 1,
        "rationale": (
            "RNA remains visually dominant within a restrained cell/nuclear context and no "
            "DNA helix, chromosome, or quantitative encoding is present."
        ),
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
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(FAMILIES),
        help="Generate only the named asset families and preserve other manifest records.",
    )
    args = parser.parse_args()
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("AZURE_OPENAI_API_KEY is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_families = set(args.families or FAMILIES)
    tasks = [
        (family, index, prompt)
        for family, prompts in FAMILIES.items()
        if family in selected_families
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

    manifest_path = args.output_dir.parent / "generation_manifest.json"
    if args.families and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        records.extend(
            record
            for record in previous.get("assets", [])
            if record.get("family") not in selected_families
        )
    records.sort(key=lambda item: (str(item["family"]), int(item["candidate"])))
    manifest = {
        "purpose": (
            "Non-quantitative illustrative components; all data marks and annotations "
            "are composed separately in vector form."
        ),
        "assets": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    curate_assets(args.output_dir)


if __name__ == "__main__":
    main()
