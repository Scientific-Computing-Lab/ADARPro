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
        "the scene sparse, biologically restrained, and unmistakably about RNA rather than DNA.",
    ],
    "pooled_human_context": [
        "A compact scientific emblem for a pooled human-tissue endpoint. Arrange four refined "
        "miniature tissue cues—arterial vessel cross-section, brain, liver, and skeletal muscle—"
        "as balanced quadrants within one restrained circular analytical field. Equal visual weight, "
        "coherent silhouette, no fifth tissue, no mixing vessel, no mathematical symbol, and no text.",
        "A polished genomics-figure asset representing aggregation across four human tissues: artery, "
        "brain, liver, and skeletal muscle appear as four recognizable but subtle sectors of one unified "
        "round mosaic. The composition communicates pooling without implying a biological mixture. No "
        "arrows, sigma symbol, labels, numbers, test tube, or clinical iconography.",
        "An elegant circular endpoint glyph composed of four equally weighted biomedical motifs: vessel "
        "wall, cerebral cortex, liver lobe, and striated muscle. Integrate the motifs through a shared thin "
        "boundary and restrained palette while keeping each recognizable. No text, equations, arrows, "
        "sample container, pie-chart percentages, or decorative badge treatment.",
    ],
    "whole_duplex_graph_view": [
        "A compact scientific representation of a whole RNA duplex as a sequence–structure graph. Show "
        "one imperfect folded RNA substrate with fine backbone contours and base-pair contacts, overlaid "
        "with a restrained translucent network of local adjacency and pairing connections. The RNA remains "
        "primary; the graph overlay is subtle and analytical. No arrows, labels, boxes, DNA helix, or chart.",
        "A high-end editorial insert showing the global computational view of an imperfect dsRNA substrate. "
        "Render a continuous folded RNA contour with several paired contacts and a small bulge; integrate "
        "delicate nodes at nucleotides and two classes of hairline connections along and across the stem. "
        "Avoid a generic node cloud, DNA icon, architecture diagram, arrows, and text.",
        "A refined molecular-computation vignette: a horizontal imperfect RNA duplex translated into a "
        "structure-aware graph through small backbone nodes and elegant pairing arcs embedded directly in "
        "the molecule. Use a compact low-profile silhouette and subtle depth. No disconnected network, "
        "regular ladder, DNA twist, labels, equations, or pipeline arrows.",
        "A horizontal analytical portrait of one continuous RNA strand folded back into two uneven, nearly "
        "parallel arms. Join the arms with sparse dotted pairing contacts, include one bulge and one loop, "
        "and place tiny graph nodes directly on the backbone with curved adjacency and cross-arm edges. The "
        "shape must be an irregular planar RNA fold, never a spiral helix, DNA ladder, or detached node cloud.",
        "A refined sequence-structure graph embedded in a folded RNA substrate: one charcoal-and-teal RNA "
        "contour bends into a broad U and returns as an offset paired arm, with subtle nucleotide nodes, "
        "backbone edges, and a few cross-strand pairing arcs. Low-profile horizontal silhouette, no twist, "
        "no canonical helix, no colored base bars, no separate network, and no labels.",
        "A PNAS-style computational-biology insert showing an imperfect RNA hairpin as an integrated graph. "
        "Use two softly curved horizontal backbone tracks connected by delicate nonuniform pairing contacts; "
        "embed small analytical nodes and hairline edges in the tracks. Include asymmetry and a small bulge. "
        "Absolutely no corkscrew, DNA double helix, regular ladder, disconnected mesh, arrows, or text.",
    ],
    "pair_weighting_editorial": [
        "A compact conceptual scientific asset for equal weighting of RNA substrates. Show two distinct "
        "imperfect RNA duplexes side by side: one bears only a few small site marks and the other bears many, "
        "yet each rests on an equal-size restrained contribution pedestal. Symmetric composition without a "
        "literal balance scale. No arrows, labels, numbers, boxes, charts, or quantitative values.",
        "A polished editorial comparison of two RNA duplex substrates with unequal densities of tiny target "
        "marks but equal visual emphasis as independent units. Use matching circular support halos beneath "
        "both substrates to communicate equal contribution. No balance-scale clipart, arrows, mathematical "
        "symbols, text, or implied measured effect size.",
        "A small publication-quality concept for pair-first learning: two separate folded RNA substrates, "
        "one site-rich and one site-sparse, are presented as equally prominent biological units in a balanced "
        "two-column composition. Fine scientific linework, no weighing scale, no arrows, labels, numerals, "
        "data plot, or decorative background.",
        "Two separate planar folded-RNA substrates in a balanced horizontal composition. Each is one continuous "
        "strand folded into an irregular paired hairpin with sparse dotted contacts. The left substrate has three "
        "small amber site dots and the right has many; place both over identical thin oval support shadows to "
        "signal equal analytical weight. No helices, DNA ladders, pedestals, scales, arrows, text, or numbers.",
        "A clean editorial concept of pair-first sampling: two unequal RNA hairpin substrates, one compact and "
        "site-sparse and one longer and site-rich, occupy identical visual frames and have identical optical "
        "weight. Render the RNA as irregular near-planar fold-back contours with a few pairing dashes. Avoid "
        "DNA twists, laboratory stands, balance-scale metaphors, boxes, labels, and quantitative decoration.",
        "A publication-quality paired-substrate vignette with two distinct imperfect dsRNA fold-backs side by "
        "side, both drawn at equal prominence. Mark several candidate sites on each, with visibly different "
        "site counts but no implied outcome. Use elegant asymmetric RNA contours and delicate pairing cues; no "
        "canonical double helices, platforms, arrows, equations, labels, or charts.",
    ],
    "target_local_editorial": [
        "A compact target-local RNA representation for a scientific figure. Show an imperfect RNA duplex "
        "with one central amber adenosine paired to one teal partner; a refined translucent optical aperture "
        "isolates a short local neighborhood while the flanking duplex softly recedes. No magnifying-glass "
        "handle, arrows, brackets, labels, DNA helix, or toy-like geometric blocks.",
        "A high-end molecular-biology vignette of target-centered local information: a short A-form RNA stem "
        "with one amber focal nucleotide and its directly opposed teal partner, surrounded by a crisp local "
        "window and softly attenuated flanks. Keep the window as subtle depth and light, not a drawn box. No "
        "text, arrows, glow, DNA iconography, or detached tokens.",
        "A refined close-focus scientific illustration of one paired site within an imperfect RNA stem. The "
        "central amber and teal nucleotides are structurally connected and sharply resolved; nearby bases "
        "remain visible, while the distant substrate fades gently. Horizontal composition, no lens icon, "
        "labels, brackets, arrows, DNA twist, or decorative halo.",
        "A precise horizontal local RNA fold shown nearly flat: two gently curved backbone tracks face one "
        "another across six delicate pairing contacts. At the center, one amber nucleotide is directly opposite "
        "one teal nucleotide; the two outermost contacts fade softly. Include one small bulge. No helix twist, "
        "DNA ladder, circular lens, glow, geometric molecule tokens, arrows, or text.",
        "A high-end editorial close-up of a target-centered RNA neighborhood. One continuous RNA contour folds "
        "into a short asymmetric stem; an amber target on one arm faces a teal partner on the other, connected "
        "by a subtle dotted pairing cue. Keep the immediate neighborhood crisp and the transcript tails faint. "
        "No spiral helix, atom model, magnifying glass, border, labels, or callout symbols.",
        "A compact planar dsRNA detail suitable for a journal architecture figure: two uneven antiparallel RNA "
        "arms, five sparse pairing dashes, a central amber target aligned to a teal partner, and a single nearby "
        "mismatch. Fine scientific linework with restrained depth. Never depict a corkscrew, DNA helix, regular "
        "ladder, nucleobase chemistry diagram, lens, arrows, or text.",
    ],
    "substrate_context_editorial": [
        "A compact multiscale biology asset linking RNA substrate organization to cellular context. Place one "
        "imperfect folded RNA transcript sharply in the foreground, its long single-stranded path extending "
        "toward a restrained translucent mammalian cell and nucleus behind it. Include sparse generic RNA-"
        "binding regulator forms. No DNA, chromosome, chromatin fibers, arrows, labels, or causal mechanism.",
        "A polished PNAS-style concept of substrate and context: one continuous transcript forms an imperfect "
        "dsRNA stem, continues through a long loop, and sits within a softly indicated cell/nuclear environment "
        "with a few generic regulator silhouettes. RNA must dominate. No DNA helix, organelle catalogue, "
        "arrows, labels, or quantitative encoding.",
        "A refined scientific vignette spanning transcript organization and cellular state. Show a paired RNA "
        "substrate connected to a longer transcript contour, with a subtle cell boundary, nuclear field, and "
        "two generic protein shapes behind it. Preserve spatial layering without implying causality. Exclude "
        "DNA, chromosomes, text, arrows, charts, and decorative scenery.",
        "A compact RNA-centered cellular-context asset: a soft partial mammalian cell and nucleus sit behind one "
        "dominant orange transcript that forms a small irregular fold-back stem with dotted pairing contacts. "
        "Add two restrained generic RNA-binding shapes and one faint alternative transcript path. The RNA must "
        "be a flexible contour, not a helix. No DNA, chromosomes, chromatin, labels, arrows, or organelle collage.",
        "A refined transcript-state vignette on transparency. Show one long single-stranded RNA path passing "
        "through a subtle nuclear field, with an asymmetric local hairpin, a small bulge, two translucent isoform "
        "branches, and sparse protein silhouettes. Keep the cell boundary secondary and the RNA unmistakable. "
        "No regular double helix, genomic material, transcription machinery, arrows, symbols, or text.",
        "A PNAS-style conceptual illustration of unresolved matched cellular context: a continuous RNA contour "
        "with one imperfect paired fold occupies the foreground; behind it are a quiet cell/nuclear silhouette, "
        "two alternative transcript routes, and softly varying generic regulators. Preserve biological restraint. "
        "Exclude DNA-like twists, chromosomes, chromatin strands, labels, arrows, question marks, and scenery.",
    ],
    "complementary_views_editorial": [
        "A compact scientific concept for complementary global and local views of the same RNA substrate. "
        "Show one complete imperfect RNA duplex and, beside it, a refined close-up of its amber/teal target "
        "pair, joined visually by shared color and a subtle overlapping analytical field. No arrows, labels, "
        "equations, boxes, lens handle, or final prediction symbol.",
        "A polished editorial asset showing two coordinated representations of one RNA: a full folded duplex "
        "and a target-centered paired-site detail, balanced in parallel and unified by a restrained central "
        "overlap. The two views must remain distinct but clearly related. No arrows, text, plus signs, network "
        "pipeline, or quantitative marks.",
        "A high-end molecular information-flow vignette with a whole RNA substrate on one side and an enlarged "
        "local target/partner neighborhood on the other. Use matching backbone colors and a subtle shared "
        "contour to communicate complementarity without a causal arrow. No labels, boxes, equations, DNA "
        "helix, or decorative glow.",
        "Two coordinated scientific views of one planar folded RNA substrate arranged side by side: a complete "
        "irregular fold-back with sparse pairing contacts, and a close target-centered detail with an amber site "
        "facing a teal partner. Use the same backbone palette and matching curvature so their relation is clear. "
        "No twisted helices, DNA ladders, magnifying glass, arrows, plus sign, labels, boxes, or equations.",
        "A refined paired visual for global and local RNA representations. On the left, show the entire low-profile "
        "asymmetric RNA hairpin; on the right, show only its local paired neighborhood at larger scale, centered "
        "on amber and teal nucleotides. Balance both views on one baseline with open space between. No DNA twist, "
        "lens cone, pipeline connector, text, symbols, glow, or decorative background.",
        "A publication-grade molecular information asset containing two separate but stylistically matched RNA "
        "illustrations: one whole imperfect duplex and one cropped target/partner neighborhood. Render both as "
        "gently curved near-planar RNA contours with delicate pairing cues. No spiral helices, DNA iconography, "
        "magnifier, arrows, borders, equations, labels, or protein complexes.",
    ],
    "global_graph_encoder_editorial": [
        "A compact horizontal interior cue for an RNA structure-graph encoder. On the left, show a small "
        "imperfect planar RNA fold with nucleotide nodes and backbone/pairing edges; through the center, "
        "show two restrained layers of neighborhood aggregation as nested translucent graph fields; on the "
        "right, resolve into one concise latent feature motif. The composition must communicate structure-aware "
        "message passing without a surrounding box, pipeline arrow, computer chip, neural-network cliché, DNA "
        "helix, text, or mathematical symbol.",
        "A refined computational-biology module interior showing a whole RNA sequence-structure graph becoming "
        "a compact learned representation. Use one irregular folded RNA graph at left, a central field of linked "
        "local neighborhoods, and a small ordered feature bundle at right. Keep the horizontal silhouette simple "
        "and legible at thumbnail scale. No container, badge, arrows, labels, circuitry, brain icon, DNA, or chart.",
        "A publication-grade visual cue for global structure-graph encoding: a near-planar imperfect RNA duplex "
        "with embedded nodes enters a quiet sequence of two graph-aggregation layers and emerges as a restrained "
        "set of aligned latent marks. Use continuity of form rather than arrows to imply processing. No UI panel, "
        "processor chip, text, equations, regular ladder, DNA twist, or decorative vertical bars.",
    ],
    "graph_local_encoder_editorial": [
        "A compact horizontal interior cue for an RNA graph-plus-local encoder. At the left, show the same whole-"
        "duplex structure graph together with a clearly distinct target-centered local RNA segment containing one "
        "amber target and teal partner; let both representations meet in a restrained shared aggregation field and "
        "resolve into one latent feature motif at right. No surrounding box, arrows, text, plus sign, DNA helix, "
        "computer chip, or generic neural-network icon.",
        "A refined computational-biology module interior that visibly combines two inputs: an irregular global RNA "
        "sequence-structure graph and a short aligned local target/partner view. Preserve blue for the global view "
        "and violet/amber/teal accents for the local view; merge them through one compact analytical field into a "
        "single ordered representation. No badge, container, pipeline arrow, label, equation, DNA, or circuitry.",
        "A publication-grade cue for graph and target-local encoding. Place a miniature whole RNA graph above a "
        "short target-centered paired neighborhood, then integrate their contours into two subtle message-passing "
        "layers and one compact latent bundle. Make the added local stream unmistakable while retaining the same "
        "visual grammar as a global graph encoder. No text, arrows, boxes, processor icon, DNA twist, or chart.",
    ],
    "prediction_integration_editorial": [
        "A compact scientific concept for computational prediction integration. Show two distinct but coordinated "
        "RNA information views on the left: one complete structure-aware RNA graph and one target-centered local "
        "paired-site detail. Their visual paths narrow toward one restrained analytical readout motif on the right, "
        "without arrows, equations, text, geometric molecule tokens, boxes, or biological fusion machinery. The "
        "meaning is two predictive representations combined into one result, not molecular interaction.",
        "A refined PNAS-style miniature showing whole-duplex and target-local representations as two parallel, "
        "scientifically consistent RNA motifs that resolve into one compact prediction readout. Keep the inputs "
        "visibly separate until the final shared analytical field. No arrows, plus signs, sigma, gauges, text, UI "
        "container, protein complex, DNA iconography, or mechanistic implication.",
        "A publication-quality computational integration asset: an imperfect full RNA graph and a cropped focal "
        "target/partner neighborhood occupy two aligned lanes, then converge spatially into one subtle neutral "
        "feature/readout mark. Use a single coherent molecular-illustration style and preserve open space for vector "
        "connectors. No labels, arrows, equation symbols, decorative polygons, network cloud, DNA helix, or chart.",
        "A thin horizontal scientific asset with exactly two input lanes and one output. The upper lane is one "
        "complete irregular planar RNA sequence-structure graph; the lower lane is one short target-centered local "
        "RNA pairing motif with an amber site opposite a teal partner. Keep the lanes separate on the left and let "
        "them end near one small neutral analytical readout mark on the right, leaving transparent gaps for vector "
        "connectors. No arrows, text, plus sign, DNA helix, protein, equation, node cloud, or decorative shapes.",
        "A compact two-view prediction vignette designed for a tiny journal panel. Stack a refined whole-RNA graph "
        "above a refined paired-site local view; align both toward one restrained set of three neutral output marks "
        "at the far right. Use the same fine editorial linework in both lanes and abundant transparency. Do not merge "
        "the molecules biologically. No pipeline arrows, labels, symbols, boxes, twisted helix, chart, or UI styling.",
        "A publication-grade miniature of computational integration, not molecular interaction. At left, show two "
        "parallel analytical portraits of the same RNA: a full imperfect fold with embedded graph nodes above and a "
        "cropped target/partner neighborhood below. At right, place one small neutral latent-output motif, separated "
        "from both inputs by open space for later vector arrows. No text, equation, sigma, convergence lines, DNA "
        "twist, protein complex, random network cloud, gauge, or decorative polygon.",
    ],
    "observed_editing_readout": [
        "A compact scientific observation motif for measured endogenous A-to-I RNA editing. Show a short imperfect "
        "RNA duplex with one focal amber adenosine and, beside it, a restrained analytical readout made of a few "
        "ordered measurement marks that clearly belong to the RNA observation. Keep the RNA and readout visually "
        "integrated but do not depict an enzyme, reaction mechanism, protein, arrow, letter, gauge, chart axis, box, "
        "DNA helix, or clinical-device icon.",
        "A refined endpoint illustration for observed RNA editing: one concise target-centered dsRNA segment with "
        "a highlighted focal nucleotide sits within a subtle measurement aperture, accompanied by a minimal row of "
        "sampled signal marks. It must read as a measured RNA state, not a process mechanism or software control. No "
        "text, arrows, equation, button, border, protein, DNA, or instrument dashboard.",
        "A publication-grade readout vignette centered on one endogenous RNA paired site. Render the local imperfect "
        "RNA context crisply, emphasize one amber target, and add a small neutral analytical trace or dot pattern that "
        "signals observation without encoding quantitative values. Use an open compact silhouette. No labels, causal "
        "arrows, enzyme, chemical reaction, rectangular UI card, DNA helix, microscope, or chart frame.",
    ],
    "search_substrate_scale": [
        "A publication-grade scientific vignette for the intrinsic substrate scale of endogenous A-to-I "
        "RNA editing. Show one continuous RNA transcript folding into an imperfect, elongated double-stranded "
        "RNA substrate with subtle nucleotide nodes and predicted pairing contacts. Include one small bulge and "
        "a restrained cue that the two arms arose from separated transcript regions. Keep the silhouette compact "
        "and horizontal. No DNA double helix, chromosome, protein, labels, arrows, chart, or mechanistic claim.",
        "A refined molecular-biology illustration of whole-duplex suitability: two regions of one RNA strand form "
        "an irregular A-form duplex with mismatches, a loop, and delicate sequence-structure graph cues embedded "
        "along the backbone. The image should communicate an RNA substrate and its organization at a glance, with "
        "a low-profile editorial composition. No DNA iconography, enzyme, text, arrows, scales, or data encoding.",
        "A high-end PNAS-style cutaway of an inverted-repeat RNA substrate. One continuous pre-mRNA bends into two "
        "paired arms; the paired region contains fine backbone and cross-arm contacts, slight asymmetry, and one "
        "bulged base. Preserve open transparent margins for vector labels. Avoid canonical DNA helices, rigid ladders, "
        "chromosomes, proteins, quantitative marks, and decorative scenery.",
    ],
    "search_context_scale": [
        "A restrained publication-quality vignette for transcript and cellular context in endogenous RNA editing. "
        "Inside a translucent mammalian cell and nucleus, show one dominant continuous RNA transcript with a small "
        "imperfect fold-back duplex, two faint alternative transcript paths, and sparse generic regulator shapes at "
        "different proximities. The image must read as unresolved matched RNA/cellular state. No DNA, chromosomes, "
        "chromatin fibers, labels, arrows, gauges, or causal mechanism.",
        "A refined scientific illustration of context-dependent RNA substrate realization: a simple cell boundary and "
        "nuclear region contain one RNA transcript whose local duplex is crisp while alternative processing and generic "
        "regulator cues fade softly around it. Keep RNA visually dominant and the composition compact. No DNA helix, "
        "transcription machinery, labels, question marks, charts, or decorative organelle catalogue.",
        "A PNAS-style conceptual asset showing the same RNA substrate embedded in a layered cellular environment. "
        "Include one continuous transcript, a small imperfect dsRNA fold, translucent isoform branches, and a few "
        "abstract RNA-binding protein forms. Use subtle depth to suggest context without encoding a measured value. "
        "No genomic DNA, chromosomes, arrows, text, causal interaction, or photorealism.",
    ],
    "search_target_scale": [
        "A precise target-local scientific illustration for A-to-I RNA editing. Show a short imperfect RNA duplex in "
        "nearly planar view, centered on one amber focal adenosine opposite a teal predicted partner. Flanking paired "
        "nucleotides, one mismatch, and one small bulge define the local neighborhood; the central pair must be obvious "
        "at thumbnail size. No DNA twist, protein, labels, arrows, brackets, glow, or quantitative data.",
        "A high-end molecular-biology close-up of target selection within an RNA duplex. Center one warm focal "
        "ribonucleotide and its directly aligned cool-colored partner inside an asymmetric local stem, with five or six "
        "subtle neighboring contacts and an off-center loop. Keep generous transparent space for vector callouts. No "
        "letters, DNA ladder, enzyme, arrows, labels, atom model, or decorative effects.",
        "A compact editorial view of an aligned target window in dsRNA: two gently curved RNA backbones frame a short "
        "paired region, one central upper adenosine is amber and the paired lower nucleotide is teal, and nearby bases "
        "fade toward the edges to emphasize local selection. The structure must remain biologically schematic and RNA-"
        "specific. No text, arrow, bracket, DNA helix, protein, or numerical mark.",
    ],
    "benchmark_pair_unit_editorial": [
        "A publication-grade horizontal illustration of one complete inverted-Alu RNA substrate as one analytical unit. "
        "Show one continuous transcript folding back into an elongated, imperfect A-form RNA duplex with unequal arms, "
        "small bulges, mismatches, and short single-stranded tails. Leave three visually clear nucleotide positions along "
        "the upper paired arm where precise vector site markers can later be overlaid. Elegant low-profile composition, "
        "biologically legible at journal-thumbnail size. No DNA double helix, chromosome, protein, text, labels, arrows, "
        "numbers, brackets, chart marks, or background panel.",
        "A refined molecular-biology plate of a complete inverted-repeat RNA pair. One flexible RNA contour bends into two "
        "long antiparallel paired arms, with delicate base-pair contacts, one asymmetric internal loop, and several exposed "
        "adenosine-sized positions. Keep the structure almost planar and horizontally elongated so colored vector markers "
        "can be integrated afterward. No B-DNA twist, rigid ladder, enzyme, letters, arrows, labels, data, or scenery.",
        "A high-end editorial cutaway of one endogenous inverted-Alu dsRNA substrate. Depict two transcript-derived arms as "
        "a continuous RNA fold with fine ribose-phosphate rails, irregular paired contacts, two modest bulges, and open "
        "single-stranded ends. The complete substrate should read immediately as one coherent unit, with three uncluttered "
        "candidate-site locations distributed along the stem for later vector annotation. No DNA iconography, atom model, "
        "protein, text, arrows, labels, charts, or decorative background.",
        "A sophisticated PNAS-style RNA substrate vignette: a complete inverted-repeat transcript folds into a graceful, "
        "imperfect A-form duplex, with subtle groove depth, sparse nucleotide detail, asymmetric mismatches, and three clean "
        "site positions visible on one arm. Use restrained scientific linework and a broad horizontal silhouette that remains "
        "clear after reduction. Avoid canonical DNA helices, colorful ladder bars, proteins, labels, arrows, numbers, boxes, "
        "and ornamental effects.",
        "A continuous pre-mRNA drawn as one fine charcoal-and-teal contour folds back horizontally to form a long imperfect "
        "inverted-Alu duplex. Use delicate short pairing contacts rather than colored base blocks, preserve two asymmetric "
        "bulges and loose transcript tails, and leave three open nucleotide positions for later vector site markers. Flat "
        "editorial molecular illustration with restrained depth, no bead chain, no DNA twist, no ladder, no protein, no text, "
        "no arrows, and no decorative scene.",
        "A precise low-profile scientific illustration of one complete inverted-repeat RNA substrate, viewed almost flat. "
        "Two unequal regions of the same thin transcript contour pair through sparse hydrogen-bond dashes, interrupted by a "
        "mismatch and one small internal loop. The form should be elegant and clearly RNA, with no colored nucleotide beads or "
        "rigid bars. Leave three clear positions along the paired region for vector overlays. No DNA helix, labels, arrows, "
        "protein, chart, border, or background.",
    ],
    "development_fold_pool_editorial": [
        "A compact editorial scientific asset representing a collection of independent RNA duplex substrates. Arrange five "
        "small but refined imperfect folded-RNA silhouettes in a shallow horizontal constellation, each distinct in contour "
        "but equal in visual weight, with ample transparent space between them for later vector fold labels. No containers, "
        "boxes, arrows, numbers, text, DNA helices, charts, or implied performance values.",
        "A polished miniature cohort of five independent inverted-repeat RNA substrates laid out from left to right. Each is "
        "an elegant low-profile imperfect dsRNA fold with slightly different bulges and stem lengths; all use the same scale "
        "and scientific illustration style. Keep generous gaps for vector annotations. No labels, numerals, arrows, panels, "
        "data marks, proteins, or B-DNA imagery.",
        "Five separate RNA-duplex units in one wide publication-quality asset, designed to sit beneath a vector brace that "
        "defines pair-disjoint folds. Render each unit as a continuous transcript forming a small imperfect paired stem, not "
        "as a generic icon. Equal prominence, restrained depth, clean transparent background. No text, boxes, arrows, charts, "
        "DNA double helices, or decorative network motifs.",
        "A wide editorial field containing exactly five small independent inverted-Alu RNA substrates. Each substrate is a "
        "thin continuous transcript folded into a low horizontal imperfect duplex with sparse pairing dashes, a different "
        "minor bulge, and soft single-stranded tails. Keep all five equal in size and separated by generous blank space. No "
        "vertical hairpins, DNA helices, colored base ladders, boxes, numbers, labels, arrows, charts, or background decoration.",
        "Exactly five refined planar folded-RNA substrates arranged in one horizontal row for a journal methods figure. Each "
        "is an irregular, low-profile fold-back contour with two short paired arms and one tiny mismatch; vary the contours "
        "subtly while preserving equal visual weight. Use fine charcoal, muted teal, and very sparse amber accents. No toy "
        "icons, no U-shaped hairpins, no B-DNA twist, no rigid ladder, no text, numerals, arrows, containers, or chart marks.",
    ],
    "frozen_specification_editorial": [
        "A compact publication-grade conceptual asset for a frozen computational specification. Show a precisely aligned "
        "stack of thin analytical layers containing subtle RNA-graph, checkpoint, threshold, and hash-like visual motifs, "
        "held inside a restrained faceted crystalline envelope. It must read as a sealed reproducible specification rather "
        "than a padlock or software button. No text, letters, numbers, arrows, UI card, chart, DNA, or laboratory apparatus.",
        "A refined editorial visualization of a computational model specification fixed before evaluation: several slim "
        "coherent layers—RNA graph representation, parameter mesh, threshold trace, and provenance marks—registered into one "
        "crisp immutable bundle with a subtle cool crystalline edge. Scientific and understated, not a lock icon. No text, "
        "numerals, arrows, boxes, interface controls, data values, or biological mechanism.",
        "A high-end scientific motif for a frozen analysis specification. Depict a small ordered bundle of translucent "
        "technical sheets with consistent RNA-graph and model-state cues, enclosed by one precise cool-toned boundary that "
        "signals fixation and provenance. Compact silhouette with open space for vector arrows. Avoid padlocks, snowflakes, "
        "buttons, labels, text, equations, charts, DNA helices, or decorative geometry unrelated to analysis.",
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
    "pooled_human_context": {
        "candidate": 1,
        "rationale": (
            "The artery, brain, liver, and muscle cues remain equally weighted and "
            "recognizable within one pooled endpoint without a mathematical badge."
        ),
    },
    "whole_duplex_graph_view": {
        "candidate": 4,
        "rationale": (
            "The integrated nucleotide nodes and backbone/pairing edges follow one "
            "irregular planar RNA fold without canonical DNA-helix imagery."
        ),
    },
    "pair_weighting_editorial": {
        "candidate": 6,
        "rationale": (
            "Two distinct imperfect RNA substrates carry different site densities at "
            "matched visual weight, directly supporting the pair-first concept."
        ),
    },
    "target_local_editorial": {
        "candidate": 4,
        "rationale": (
            "The nearly planar local stem makes the central amber target and teal partner "
            "immediately legible while flanking context recedes cleanly."
        ),
    },
    "substrate_context_editorial": {
        "candidate": 6,
        "rationale": (
            "A dominant continuous RNA contour and imperfect fold sit against a restrained "
            "cellular field without DNA, chromatin, or causal iconography."
        ),
    },
    "complementary_views_editorial": {
        "candidate": 4,
        "rationale": (
            "The full folded substrate and isolated target-pair detail remain visually "
            "distinct and coordinated without an imposed processing arrow."
        ),
    },
    "global_graph_encoder_editorial": {
        "candidate": 2,
        "rationale": (
            "The sequence-structure graph, restrained aggregation field, and compact "
            "feature bundle remain distinct after reduction without UI or DNA imagery."
        ),
    },
    "graph_local_encoder_editorial": {
        "candidate": 1,
        "rationale": (
            "Separate global and focal RNA cues resolve into a shared representation, "
            "making the additional local input visible at the module's small footprint."
        ),
    },
    "prediction_integration_editorial": {
        "candidate": 4,
        "rationale": (
            "The clearest generated two-lane concept; retained for provenance, while "
            "the final miniature uses a more legible vector assembly at print scale."
        ),
    },
    "observed_editing_readout": {
        "candidate": 3,
        "rationale": (
            "A single-contour paired RNA target leads into a restrained measurement trace, "
            "reading as an observed endpoint without a DNA helix or mechanism claim."
        ),
    },
    "search_substrate_scale": {
        "candidate": 3,
        "rationale": (
            "The continuous fold-back contour and asymmetric central stem best convey "
            "substrate formation; the more DNA-like multihelix alternatives were rejected."
        ),
    },
    "search_context_scale": {
        "candidate": 1,
        "rationale": (
            "The RNA strand remains the dominant continuous object within a restrained "
            "cellular field, with no chromosome or genomic DNA motif."
        ),
    },
    "search_target_scale": {
        "candidate": 2,
        "rationale": (
            "The central opposing nucleotides and surrounding asymmetric stem remain "
            "visible at small scale; alternatives lost the paired-neighborhood context."
        ),
    },
    "benchmark_pair_unit_editorial": {
        "candidate": 6,
        "rationale": (
            "The continuous, low-profile transcript contour reads as one complete RNA "
            "substrate and leaves a clean paired region for controlled vector site markers."
        ),
    },
    "development_fold_pool_editorial": {
        "candidate": 4,
        "rationale": (
            "Five separate horizontal RNA substrates remain visually equal and compact "
            "enough to identify the pair pool without turning the fold strip into clipart."
        ),
    },
    "frozen_specification_editorial": {
        "candidate": 2,
        "rationale": (
            "The layered analytical bundle conveys a fixed specification and provenance "
            "without using a padlock, software control, or biological mechanism metaphor."
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
                raise RuntimeError(
                    f"Selected asset has an empty alpha channel: {source}"
                )
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
