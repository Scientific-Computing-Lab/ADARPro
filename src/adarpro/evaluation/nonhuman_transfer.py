"""Transfer of the frozen selected predictor to AdarEdit's nonhuman datasets, without retraining.

Question: trained only on human inverted-Alu duplexes, how well does each endpoint's frozen selected predictor
(``models/final_v2``: the genomic-spacing stage on the graph and CNN-alone networks, and the equal-weight average of
the same networks without the stage) rank editing sites in AdarEdit's three nonhuman test splits (Octopus,
Ptychodera, Strongylocentrotus), and how does it compare with AdarEdit's supplied checkpoints on the same sites?

Stages (subcommands). All but ``evaluate`` are label-free.

- ``sites``: the nonhuman validation and test rows from the pinned AdarEdit package (``data/species/<animal>/`` JSONL
  + metadata, ``data/species_prebalancing/``), joined row for row and checked. The label and editing-level columns
  are not read, and the JSONL label is discarded as each line is parsed.
- ``score``: every row through each endpoint's frozen predictor (checkpoints bound by hash), labels set to 0.
- ``adaredit``: AdarEdit's per-site test probabilities from its 48 prediction archives (``results/preds/``; 2
  checkpoints x 8 training sources x 3 animals). Only the ``probs`` array of each archive is loaded.
- ``evaluate``: the test labels; checks before any metric (JSONL = metadata labels; each archive's labels = the JSONL
  labels line by line; AdarEdit's AUROC and F1 recomputed = its published matrices), then AUROC, AUPRC, F1 and paired
  differences in AUROC.

Genomic gap: the number of bases strictly between the two paired arms, the convention the spacing stage was fitted
on (``gap_stage.pair_gap_table``: ``start2 - end1`` on BED half-open coordinates). AdarEdit's arm coordinates are
1-based inclusive, so they are converted to BED (start - 1) and the gap comes from the same function
(``genomic_context.pair_intrinsic_features``).

Comparators: for each training source and animal, the stronger of AdarEdit's two supplied checkpoints is the one with
the higher published AUROC (``results/matrix_{baseline,bioaware}_auroc.csv``). AdarEdit trained on the animal itself
is chosen the same way.

Intervals: paired bootstraps that resample whole genomic regions (AdarEdit's split unit), 2,000 replicates, seed
20261117, one set of region draws per animal shared by every endpoint and model. "BEATS" / "WORSE" mean the 95%
interval of the difference excludes zero; otherwise the two cannot be told apart. The intervals condition on the
fitted models.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adarpro.quarantine import CONTEXTS, sha256_file

ANIMALS = ("Octopus", "Ptychodera", "Strongylocentrotus")
SPLITS = ("valid", "test")
EXPECTED_ROWS = {("Octopus", "test"): 1678, ("Ptychodera", "test"): 1634, ("Strongylocentrotus", "test"): 1493,
                 ("Octopus", "valid"): 1242, ("Ptychodera", "valid"): 1213, ("Strongylocentrotus", "valid"): 1268}
ADAREDIT_VARIANTS = ("baseline", "bioaware")  # the supplied compact and Bio-aware checkpoints
ADAREDIT_SOURCES = ("Artery", "Brain", "Combined", "Liver", "MuscleSkeletal") + ANIMALS
NB, SEED = 2000, 20261117
BATCH_SIZE = 256
THREADS = 8
OUTPUT = Path("results/adarpro/nonhuman_transfer")
KEYS = ["species", "split", "line_no", "region_id", "molecule_key"]
METADATA_COLUMNS = ["species", "region_id", "source_row_id", "split", "Chr", "Position", "Strand", "Local_Position"]
PREBALANCING_COLUMNS = ["Chr", "Position", "Strand", "Local_Position", "start_cluster", "end_cluster", "small_ds_seq",
                        "mfe_struct", "start_1ds_genome", "end_1ds_genome", "start_2ds_genome", "end_2ds_genome"]
# (a, b, role): each contrast is a - b in AUROC. Primary: against AdarEdit's stronger human-trained checkpoint of the
# same endpoint, and the spacing stage against the same networks without it.
CONTRASTS = [
    ("selected", "adaredit_human_stronger", "primary"),
    ("selected", "no_gap", "primary"),
    ("selected", "adaredit_human_baseline", "descriptive"),
    ("selected", "adaredit_animal_stronger", "descriptive"),
    ("no_gap", "adaredit_human_stronger", "descriptive"),
    ("no_gap", "adaredit_human_baseline", "descriptive"),
    ("no_gap", "adaredit_animal_stronger", "descriptive"),
]
HEADLINE_ENDPOINT = "Artery"


class Checks:
    """Every check is recorded; ``require`` raises on the first failure."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def __call__(self, name: str, expected, observed, ok: bool) -> bool:
        self.rows.append({"check": name, "expected": str(expected), "observed": str(observed), "pass": bool(ok)})
        print("{} {} | expected={} observed={}".format("PASS" if ok else "FAIL", name, expected, observed), flush=True)
        return bool(ok)

    def require(self, what: str) -> None:
        failed = [row["check"] for row in self.rows if not row["pass"]]
        if failed:
            raise RuntimeError(f"{what}: {len(failed)} failed check(s): {failed}")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["check", "expected", "observed", "pass"])


# ---------------------------------------------------------------- statistics (resampling whole regions)


def boot_weights(regions: np.ndarray, rng: np.random.Generator, replicates: int = NB) -> np.ndarray:
    """Row weights for ``replicates`` region resamples; row 0 is the unweighted sample (all ones)."""

    unique, inverse = np.unique(regions, return_inverse=True)
    weights = np.ones((replicates + 1, len(regions)))
    for b in range(1, replicates + 1):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights[b] = counts[inverse]
    return weights


def wauc(score: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted AUROC for each weight row of W (ties count one half). W: (B, n) nonnegative."""

    order = np.argsort(score, kind="mergesort")
    s, yy, Ws = score[order], y[order], W[:, order]
    _, start = np.unique(s, return_index=True)
    grp = np.repeat(np.arange(len(start)), np.diff(np.append(start, len(s))))
    neg = Ws * (yy == 0)
    pos = Ws * (yy == 1)
    G = len(start)
    neg_g = np.zeros((W.shape[0], G))
    pos_g = np.zeros((W.shape[0], G))
    np.add.at(neg_g.T, grp, neg.T)
    np.add.at(pos_g.T, grp, pos.T)
    neg_below = np.cumsum(neg_g, axis=1) - neg_g
    num = (pos_g * (neg_below + 0.5 * neg_g)).sum(axis=1)
    return num / (pos_g.sum(axis=1) * neg_g.sum(axis=1))


def word(lo: float, hi: float) -> str:
    return "BEATS" if lo > 0 else "WORSE" if hi < 0 else "cannot be told apart (NOT SHOWN)"


def best_f1(y: np.ndarray, s: np.ndarray) -> float:
    """Best F1 over all cut-offs ``s >= t`` (an oracle ceiling; it uses the labels)."""

    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    tp = np.cumsum(yy)[last]
    called = last + 1
    return float(np.max(2 * tp / (called + yy.sum())))


# ---------------------------------------------------------------- inputs


def bed_gap(lo_start: int, lo_end: int, hi_start: int, hi_end: int) -> tuple[int, int, int, int]:
    """1-based inclusive arm coordinates (lower arm first) -> BED half-open (start - 1, end)."""

    return lo_start - 1, lo_end, hi_start - 1, hi_end


def pinned_hashes(repository: Path) -> dict[str, str]:
    pins = pd.read_csv(repository / "provenance/excluded_artifacts.csv", usecols=["path", "source_sha256"])
    return dict(zip(pins["path"], pins["source_sha256"]))


def site_table(repository: Path, adaredit_root: Path, check: Checks) -> pd.DataFrame:
    """Label-free nonhuman validation + test rows, in a fixed order (animal, split, JSONL line)."""

    import sys

    from adarpro.features.genomic_context import pair_intrinsic_features
    from adarpro.models.baseline_oof import upstream_modules

    sys.dont_write_bytecode = True
    baseline = upstream_modules(repository)
    pin = pinned_hashes(repository)
    frames = []
    for species in ANIMALS:
        rel = f"data/species_prebalancing/{species}_prebalancing.csv.gz"
        check(f"sites {species}: prebalancing file sha256 == the release pin", pin.get(rel)[:12],
              sha256_file(adaredit_root / rel)[:12], sha256_file(adaredit_root / rel) == pin.get(rel))
        pre = pd.read_csv(adaredit_root / rel, usecols=PREBALANCING_COLUMNS, dtype=str, keep_default_na=False)
        pre.index = np.arange(1, len(pre) + 1)  # AdarEdit's source_row_id is 1-based
        for split in SPLITS:
            jsonl = f"data/species/{species}/{split}.jsonl"
            meta = f"data/species/{species}/{split}.metadata.csv"
            for rel in (jsonl, meta):
                digest = sha256_file(adaredit_root / rel)
                check(f"sites {species}/{split}: {Path(rel).name} sha256 == the release pin", pin.get(rel)[:12],
                      digest[:12], digest == pin.get(rel))
            parsed = []
            with (adaredit_root / jsonl).open() as handle:
                for line_no, line in enumerate(handle):
                    sequence, structure, target, _label = baseline.parse_openai_json(json.loads(line))
                    del _label  # label-free: discarded on the spot
                    parsed.append((line_no, sequence, structure, int(target)))
            p = pd.DataFrame(parsed, columns=["line_no", "sequence_raw", "structure", "local_position"])
            md = pd.read_csv(adaredit_root / meta, usecols=METADATA_COLUMNS, dtype=str, keep_default_na=False)
            n = len(p)
            check(f"sites {species}/{split}: rows", EXPECTED_ROWS[(species, split)], f"{n} JSONL, {len(md)} metadata",
                  n == len(md) == EXPECTED_ROWS[(species, split)])
            check(f"sites {species}/{split}: metadata species and split", f"{species}/{split}",
                  f"{sorted(md.species.unique())}/{sorted(md.split.unique())}",
                  set(md.species) == {species} and set(md.split) == {split})
            p["sequence"] = p["sequence_raw"].str.replace("U", "T")
            check(f"sites {species}/{split}: alphabet after U->T", "subset of ACGTN",
                  "".join(sorted(set("".join(p.sequence)))), set("".join(p.sequence)) <= set("ACGTN"))
            check(f"sites {species}/{split}: target base is A", 0,
                  int(sum(s[i] != "A" for s, i in zip(p.sequence, p.local_position))),
                  all(s[i] == "A" for s, i in zip(p.sequence, p.local_position)))
            check(f"sites {species}/{split}: sequence length == structure length", 0,
                  int((p.sequence.str.len() != p.structure.str.len()).sum()),
                  bool((p.sequence.str.len() == p.structure.str.len()).all()))
            check(f"sites {species}/{split}: Local_Position == target + 1 (1-based)", 0,
                  int((md.Local_Position.astype(int).to_numpy() != p.local_position.to_numpy() + 1).sum()),
                  bool((md.Local_Position.astype(int).to_numpy() == p.local_position.to_numpy() + 1).all()))
            sid = md.source_row_id.astype(int)
            joined = bool(sid.isin(pre.index).all())
            check(f"sites {species}/{split}: every source_row_id joins a prebalancing row", n, int(sid.isin(pre.index).sum()),
                  joined)
            if not joined:
                raise RuntimeError(f"{species}/{split}: unjoined source rows")
            src = pre.loc[sid.to_numpy()].reset_index(drop=True)
            for column in ("Chr", "Position", "Strand", "Local_Position"):
                bad = int((src[column].to_numpy() != md[column].to_numpy()).sum())
                check(f"sites {species}/{split}: {column} agrees with the prebalancing row", 0, bad, bad == 0)
            # AdarEdit's JSONL shows A at the target; a few source rows carry another base there (described, not
            # an error), so the sequences are compared with the source's target base set to A.
            source_seq = src.small_ds_seq.str.upper().to_numpy()
            target_not_a = int(sum(s[i] != "A" for s, i in zip(source_seq, p.local_position)))
            source_seq = [s[:i] + "A" + s[i + 1:] for s, i in zip(source_seq, p.local_position)]
            bad = int((np.asarray(source_seq) != p.sequence_raw.to_numpy()).sum())
            check(f"sites {species}/{split}: sequence == prebalancing small_ds_seq (target base as A; source "
                  f"target base not A in {target_not_a} rows)", 0, bad, bad == 0)
            bad = int((src.mfe_struct.to_numpy() != p.structure.to_numpy()).sum())
            check(f"sites {species}/{split}: structure == prebalancing mfe_struct", 0, bad, bad == 0)
            region = src.Chr + ":" + src.Strand + ":" + src.start_cluster + ":" + src.end_cluster
            bad = int((region.to_numpy() != md.region_id.to_numpy()).sum())
            check(f"sites {species}/{split}: region_id == Chr:Strand:cluster", 0, bad, bad == 0)
            # arms, genome-sorted (lower arm first); source coordinates are 1-based inclusive
            s1, e1 = src.start_1ds_genome.astype(int).to_numpy(), src.end_1ds_genome.astype(int).to_numpy()
            s2, e2 = src.start_2ds_genome.astype(int).to_numpy(), src.end_2ds_genome.astype(int).to_numpy()
            one_lower = s1 < s2
            lo_s, lo_e = np.where(one_lower, s1, s2), np.where(one_lower, e1, e2)
            hi_s, hi_e = np.where(one_lower, s2, s1), np.where(one_lower, e2, e1)
            strand = md.Strand.to_numpy()
            lo_len, hi_len = lo_e - lo_s + 1, hi_e - hi_s + 1
            first = np.where(strand == "+", lo_len, hi_len)
            linker = np.array([s[f:f + 10] == "N" * 10 for s, f in zip(p.sequence, first)])
            check(f"sites {species}/{split}: the 10-N linker sits where the arm coordinates put it", n,
                  int(linker.sum()), bool(linker.all()))
            second = np.where(strand == "+", hi_len, lo_len)
            arms_ok = (p.sequence.str.len().to_numpy() == first + 10 + second)
            check(f"sites {species}/{split}: arm lengths + linker == molecule length", n, int(arms_ok.sum()),
                  bool(arms_ok.all()))
            bed = np.array([bed_gap(*row) for row in zip(lo_s, lo_e, hi_s, hi_e)])
            gap_bp, gap = [], []
            for (b1, c1, b2, c2), structure, sequence, sd in zip(bed, p.structure, p.sequence, strand):
                features = pair_intrinsic_features(pd.Series({"structure": structure, "sequence": sequence,
                                                              "strand": sd, "start1": b1, "end1": c1, "start2": b2,
                                                              "end2": c2}))
                gap_bp.append(features["geometry__genomic_gap"])
                gap.append(features["geometry__genomic_gap_log1p"])
            p["gap_bp"], p["gap"] = gap_bp, gap
            check(f"sites {species}/{split}: gap == bases strictly between the arms (hi_start - lo_end - 1)",
                  "equal", "", bool(np.array_equal(p.gap_bp.to_numpy(), (hi_s - lo_e - 1).astype(float))))
            check(f"sites {species}/{split}: no negative gap", 0, int((p.gap_bp < 0).sum()), bool((p.gap_bp >= 0).all()))
            p["chr"], p["strand"] = md.Chr.to_numpy(), strand
            p["start1"], p["end1"], p["start2"], p["end2"] = bed[:, 0], bed[:, 1], bed[:, 2], bed[:, 3]
            p["region_id"] = md.region_id.to_numpy()
            p.insert(0, "split", split)
            p.insert(0, "species", species)
            frames.append(p)
    sites = pd.concat(frames, ignore_index=True)
    sites["molecule_key"] = pd.util.hash_pandas_object(sites[["sequence", "structure"]], index=False).astype("uint64")
    per_molecule = sites.groupby(["species", "molecule_key"]).gap_bp.nunique()
    check("sites: one gap per molecule", 0, int((per_molecule > 1).sum()), bool((per_molecule == 1).all()))
    regions = sites.groupby(["species", "region_id"]).split.nunique()
    check("sites: no region in both validation and test", 0, int((regions > 1).sum()), bool((regions == 1).all()))
    return sites[["species", "split", "line_no", "region_id", "molecule_key", "sequence", "sequence_raw", "structure",
                  "local_position", "chr", "strand", "start1", "end1", "start2", "end2", "gap_bp", "gap"]]


# ---------------------------------------------------------------- scoring (label-free)


def score_rows(repository: Path, specification: dict, table: pd.DataFrame, gap, device, batch_size: int = BATCH_SIZE):
    """One endpoint's frozen predictor on ``table`` (example_id, context, sequence, structure, local_position)."""

    from adarpro.evaluation.final_locked import candidate_predictions
    from adarpro.models.baseline_oof import upstream_modules
    from adarpro.models.final_gap_stage import apply_frozen_stage
    from adarpro.models.gap_stage import average_probabilities

    gap = np.asarray(gap, dtype=np.float64)
    if len(gap) != len(table) or not np.isfinite(gap).all():
        raise RuntimeError("gap must be finite, one value per row")
    t = table.reset_index(drop=True).copy()
    if t["context"].nunique() != 1:
        raise RuntimeError("one endpoint per call")
    t["label"] = 0
    baseline = upstream_modules(repository)
    graphs = [baseline.create_rna_graph(r.sequence, r.structure, int(r.local_position), 0)
              for r in t.itertuples(index=False)]
    if any(g is None for g in graphs):
        raise RuntimeError(f"{sum(g is None for g in graphs)} unbuildable graphs")
    equal, outputs = candidate_predictions(
        repository, {"components": specification["components"],
                     "weights": specification["equal_average"]["weights"]}, t, graphs, device, batch_size)
    averaged = {network: average_probabilities([outputs[name] for name in names])
                for network, names in specification["networks"].items()}
    stage = json.loads((repository / specification["stage"]["path"]).read_text())
    if float(stage["threshold"]) != float(specification["threshold"]):
        raise RuntimeError("stage threshold differs from the specification's")
    stage_p = apply_frozen_stage(stage, averaged["graph"], averaged["cnn_alone"], gap)
    result = {"gap_stage": np.asarray(stage_p, dtype=np.float64), "equal_average": np.asarray(equal, dtype=np.float64),
              "graph": averaged["graph"], "cnn_alone": averaged["cnn_alone"],
              "networks": {k: np.asarray(v, dtype=np.float64) for k, v in outputs.items()}}
    for k in ("gap_stage", "equal_average", "graph", "cnn_alone"):
        if not np.isfinite(result[k]).all():
            raise RuntimeError(f"non-finite {k}")
    return result


def score_sites(repository: Path, sites: pd.DataFrame, check: Checks, threads: int = THREADS) -> pd.DataFrame:
    import torch

    from adarpro.evaluation.final_locked import weighted_logit_average
    from adarpro.evaluation.final_locked_gap_stage import context_specification
    from adarpro.models.final_gap_stage import apply_frozen_stage
    from adarpro.models.gap_stage import average_probabilities

    torch.set_num_threads(threads)
    device = torch.device("cpu")
    frames = []
    for context in CONTEXTS:
        t0 = time.time()
        spec = context_specification(repository, context)
        table = pd.DataFrame({"example_id": [f"{s}:{p}:{i}" for s, p, i in zip(sites.species, sites.split, sites.line_no)],
                              "context": context, "sequence": sites.sequence.to_numpy(),
                              "structure": sites.structure.to_numpy(),
                              "local_position": sites.local_position.to_numpy().astype(np.int64)})
        out = score_rows(repository, spec, table, sites.gap.to_numpy(np.float64), device)
        f = sites[KEYS + ["gap_bp", "gap"]].copy()
        f.insert(0, "endpoint", context)
        for column in ("gap_stage", "equal_average", "graph", "cnn_alone"):
            f[column] = out[column]
        for name in sorted(out["networks"]):
            f[name] = out["networks"][name]
        f["threshold_gap_stage"] = float(spec["threshold"])
        f["threshold_equal_average"] = float(spec["equal_average"]["threshold"])
        g = average_probabilities([f[n].to_numpy() for n in spec["networks"]["graph"]])
        c = average_probabilities([f[n].to_numpy() for n in spec["networks"]["cnn_alone"]])
        names = [component["name"] for component in spec["components"]]
        e = weighted_logit_average([f[n].to_numpy() for n in names], np.asarray(spec["equal_average"]["weights"]))
        stage = json.loads((repository / spec["stage"]["path"]).read_text())
        s = apply_frozen_stage(stage, g, c, f["gap"].to_numpy())
        check(f"score {context}: averages and stage recomputed from the stored networks == stored", "exact", "",
              np.array_equal(g, f.graph.to_numpy()) and np.array_equal(c, f.cnn_alone.to_numpy())
              and np.array_equal(e, f.equal_average.to_numpy()) and np.array_equal(s, f.gap_stage.to_numpy()))
        frames.append(f)
        print(f"score {context}: {len(f)} rows in {time.time() - t0:.0f}s", flush=True)
    scores = pd.concat(frames, ignore_index=True)
    finite = bool(np.isfinite(scores.select_dtypes("number").to_numpy(dtype=float)).all())
    check("score: all scores finite; selected predictor in (0, 1)", True, finite,
          finite and bool(((scores.gap_stage > 0) & (scores.gap_stage < 1)).all()))
    return scores


def adaredit_site_scores(adaredit_root: Path, sites: pd.DataFrame, check: Checks) -> pd.DataFrame:
    """AdarEdit's per-site test probabilities, one column per checkpoint x training source; ``labels`` not loaded."""

    test = sites[sites.split == "test"].sort_values(["species", "line_no"])
    parts = []
    for species in ANIMALS:
        t = test[test.species == species][["species", "line_no"]].reset_index(drop=True)
        columns = {}
        for variant in ADAREDIT_VARIANTS:
            for source in ADAREDIT_SOURCES:
                path = adaredit_root / "results" / "preds" / f"{variant}__train-{source}__eval-{species}.npz"
                with np.load(path, allow_pickle=False) as archive:  # lazy: only the members read below are loaded
                    ok = (str(archive["variant"]) == variant and str(archive["train_context"]) == source
                          and str(archive["eval_context"]) == species)
                    probs = archive["probs"].astype(float)
                check(f"adaredit {path.name}: fields and length", f"{variant}/{source}/{species}, n={len(t)}",
                      f"n={len(probs)}", ok and len(probs) == len(t))
                columns[f"adaredit_{variant}_{source}"] = probs
        parts.append(pd.concat([t, pd.DataFrame(columns)], axis=1))
    return pd.concat(parts, ignore_index=True)


def comparators(repository: Path) -> pd.DataFrame:
    """(training source, animal) -> the AdarEdit checkpoint with the higher published AUROC (label-free)."""

    matrices = {v: pd.read_csv(repository / f"results/matrix_{v}_auroc.csv", index_col=0) for v in ADAREDIT_VARIANTS}
    rows = []
    for species in ANIMALS:
        for source in list(CONTEXTS) + [species]:
            auroc = {v: float(matrices[v].loc[source, species]) for v in ADAREDIT_VARIANTS}
            best = max(auroc, key=auroc.get)
            rows.append({"train": source, "eval": species, "kind": "own animal" if source == species else "human",
                         "baseline_auroc": auroc["baseline"], "bioaware_auroc": auroc["bioaware"],
                         "comparator": best, "column": f"adaredit_{best}_{source}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- labels and evaluation


def test_labels(repository: Path, adaredit_root: Path, check: Checks) -> pd.DataFrame:
    """The nonhuman test labels, checked against the metadata (label and editing-level rule) before use."""

    from adarpro.models.baseline_oof import upstream_modules

    baseline = upstream_modules(repository)
    frames = []
    for species in ANIMALS:
        labels = []
        with (adaredit_root / "data/species" / species / "test.jsonl").open() as handle:
            for line_no, line in enumerate(handle):
                labels.append((species, line_no, int(baseline.parse_openai_json(json.loads(line))[3])))
        lab = pd.DataFrame(labels, columns=["species", "line_no", "label"])
        md = pd.read_csv(adaredit_root / "data/species" / species / "test.metadata.csv",
                         usecols=["label", "EditingLevel"], dtype=str, keep_default_na=False)
        meta = (md.label == "yes").astype(int).to_numpy()
        level = md.EditingLevel.astype(float).to_numpy()
        check(f"labels {species}: JSONL == metadata", len(lab), f"{int((lab.label.to_numpy() != meta).sum())} differ",
              len(md) == len(lab) and bool((lab.label.to_numpy() == meta).all()))
        rule = ((meta == 1) & (level > 0.15)) | ((meta == 0) & (level < 0.001))
        check(f"labels {species}: edited > 0.15, unedited < 0.001 (AdarEdit's rule)", len(lab), int(rule.sum()),
              bool(rule.all()))
        frames.append(lab)
    return pd.concat(frames, ignore_index=True)


def adaredit_anchor(repository: Path, adaredit_root: Path, labels: pd.DataFrame, D: pd.DataFrame,
                    check: Checks) -> None:
    from sklearn.metrics import f1_score, roc_auc_score

    worst_auc, worst_f1, aligned, n = 0.0, 0.0, 0, 0
    g = D[D.endpoint == HEADLINE_ENDPOINT]
    for variant in ADAREDIT_VARIANTS:
        Ma = pd.read_csv(repository / f"results/matrix_{variant}_auroc.csv", index_col=0)
        Mf = pd.read_csv(repository / f"results/matrix_{variant}_f1.csv", index_col=0)
        Mt = pd.read_csv(repository / f"results/matrix_{variant}_threshold.csv", index_col=0)
        for source in ADAREDIT_SOURCES:
            for species in ANIMALS:
                path = adaredit_root / "results" / "preds" / f"{variant}__train-{source}__eval-{species}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    archive_labels = archive["labels"].astype(int)
                truth = labels[labels.species == species].sort_values("line_no").label.to_numpy()
                aligned += int(np.array_equal(archive_labels, truth))
                h = g[g.species == species].sort_values("line_no")
                s, y = h[f"adaredit_{variant}_{source}"].to_numpy(float), h.label.to_numpy(int)
                worst_auc = max(worst_auc, abs(roc_auc_score(y, s) - float(Ma.loc[source, species])))
                worst_f1 = max(worst_f1, abs(f1_score(y, s >= float(Mt.loc[source, species]))
                                             - float(Mf.loc[source, species])))
                n += 1
    check(f"anchor: each archive's labels == the JSONL labels line by line ({n} archives)", n, aligned, aligned == n)
    check(f"anchor: AdarEdit AUROC recomputed == its published matrices ({n} cells)", "<= 1e-9", f"{worst_auc:.1e}",
          worst_auc <= 1e-9)
    check(f"anchor: AdarEdit F1 at its own threshold == its published matrices ({n} cells)", "<= 1e-9",
          f"{worst_f1:.1e}", worst_f1 <= 1e-9)


def analyse(D: pd.DataFrame, comparator_table: pd.DataFrame, thresholds: dict, label: str, check: Checks):
    """Metrics, F1 (description) and paired region-bootstrap AUROC differences for every endpoint x animal."""

    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    comp = {(r.train, r.eval): r.column for r in comparator_table.itertuples(index=False)}
    rng = np.random.default_rng(SEED)
    metrics, contrasts, f1rows = [], [], []
    for species in ANIMALS:
        Dsp = D[D.species == species]
        reg0 = Dsp[Dsp.endpoint == CONTEXTS[0]].sort_values("line_no").region_id.to_numpy()
        W = boot_weights(reg0, rng)
        for endpoint in CONTEXTS:
            g = Dsp[Dsp.endpoint == endpoint].sort_values("line_no")
            if not np.array_equal(g.region_id.to_numpy(), reg0):
                raise RuntimeError("rows differ across endpoints")
            y = g[label].to_numpy(int)
            models = {"selected": ("gap_stage", float(g.threshold_gap_stage.iat[0])),
                      "no_gap": ("equal_average", float(g.threshold_equal_average.iat[0])),
                      "graph_alone": ("graph", None), "cnn_alone": ("cnn_alone", None)}
            for variant in ADAREDIT_VARIANTS:
                models[f"adaredit_human_{variant}"] = (f"adaredit_{variant}_{endpoint}",
                                                       float(thresholds[variant].loc[endpoint, species]))
            for name, source in (("adaredit_human_stronger", endpoint), ("adaredit_animal_stronger", species)):
                column = comp[(source, species)]
                variant = column.split("_")[1]
                models[name] = (column, float(thresholds[variant].loc[source, species]))
            auc = {}
            for name, (column, tau) in models.items():
                s = g[column].to_numpy(float)
                auc[name] = wauc(s, y, W)
                metrics.append({"endpoint": endpoint, "species": species, "model": name, "column": column,
                                "n_sites": len(g), "n_edited": int(y.sum()), "n_regions": int(g.region_id.nunique()),
                                "auroc": float(auc[name][0]), "auprc": float(average_precision_score(y, s)),
                                "auroc_sklearn_check": float(roc_auc_score(y, s))})
                if tau is not None:
                    called = s >= tau
                    f1rows.append({"endpoint": endpoint, "species": species, "model": name, "column": column,
                                   "threshold": tau, "share_called_edited": float(called.mean()),
                                   "share_edited": float(y.mean()), "f1_own_threshold": float(f1_score(y, called)),
                                   "f1_best_on_test": best_f1(y, s)})
            for a, b, role in CONTRASTS:
                d = auc[a] - auc[b]
                lo, hi = np.percentile(d[1:], [2.5, 97.5])
                contrasts.append({"contrast": f"{a}_minus_{b}", "role": role, "endpoint": endpoint,
                                  "species": species, "a": a, "b": b, "b_column": models[b][0],
                                  "auroc_a": float(auc[a][0]), "auroc_b": float(auc[b][0]), "delta": float(d[0]),
                                  "L95": float(lo), "U95": float(hi), "word": word(lo, hi),
                                  "headline": bool(role == "primary" and endpoint == HEADLINE_ENDPOINT),
                                  "n_sites": len(g), "n_regions": int(g.region_id.nunique())})
    M, C, F = pd.DataFrame(metrics), pd.DataFrame(contrasts), pd.DataFrame(f1rows)
    gap = float(np.max(np.abs(M.auroc - M.auroc_sklearn_check)))
    check("evaluate: weighted AUROC (unit weights) == sklearn on every cell x model", "<= 1e-12", f"{gap:.1e}",
          gap <= 1e-12)
    return M, C, F


# ---------------------------------------------------------------- label-free gap summary


def human_training_gaps(repository: Path, context: str) -> pd.DataFrame:
    """Genomic gap of every human training row of one endpoint: the rows its frozen spacing stage was fitted on."""

    from adarpro.evaluation.final_locked_gap_stage import context_specification
    from adarpro.models.gap_stage import pair_gap_table

    master = pd.read_parquet(repository / "data/derived/site_master.parquet",
                             columns=["context", "source_split", "pair_id", "chr1", "chr2", "start1", "end1",
                                      "start2", "end2", "strand", "sequence", "structure"])
    train = master[(master.context == context) & (master.source_split == "train")]
    gaps = pair_gap_table(master[master.context == context]).set_index("pair_id")
    stage = json.loads((repository / context_specification(repository, context)["stage"]["path"]).read_text())
    if len(train) != int(stage["train_rows"]):
        raise RuntimeError(f"{context}: {len(train)} training rows, the frozen stage records {stage['train_rows']}")
    return pd.DataFrame({"pair_id": train.pair_id.to_numpy(), "gap_bp": gaps.loc[train.pair_id, "gap_bp"].to_numpy(),
                         "gap": gaps.loc[train.pair_id, "gap"].to_numpy()})


def gap_summary(human: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Per animal (test rows): median gap, share below the human training 1st percentile (log1p scale)."""

    hg = human.gap.to_numpy()
    p1 = float(np.percentile(hg, 1))
    rows = []
    for species in ANIMALS:
        a = sites[(sites.species == species) & (sites.split == "test")]
        rows.append({"species": species, "n_sites": len(a), "median_gap_bp": float(np.median(a.gap_bp)),
                     "share_below_human_p1": float((a.gap.to_numpy() < p1).mean()),
                     "human_rows": len(hg), "human_pairs": int(human.pair_id.nunique()),
                     "human_median_gap_bp": float(np.expm1(np.median(hg))), "human_p1_gap_bp": float(np.expm1(p1))})
    return pd.DataFrame(rows)


def gap_histogram(human: pd.DataFrame, sites: pd.DataFrame, bins: int = 43) -> pd.DataFrame:
    """Binned log1p(gap) of the human artery training sites and of each animal's test sites, for the SI figure.

    Shares are percent of all sites of the group; bins run from log1p(3 nt) to 0.1 above the largest value."""

    test = sites[sites.split == "test"]
    low = float(np.log1p(3.0))
    high = float(max(human.gap.max(), test.gap.max())) + 0.1
    edges = np.linspace(low, high, bins + 1)
    rows = []
    groups = [("human artery training", human.gap.to_numpy())]
    groups += [(species, test[test.species == species].gap.to_numpy()) for species in ANIMALS]
    for group, values in groups:
        counts, _ = np.histogram(values, bins=edges)
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            rows.append({"group": group, "bin_left": float(left), "bin_right": float(right), "sites": int(count),
                         "share_percent": 100.0 * count / len(values), "group_sites": len(values),
                         "group_sites_below_range": int((values < low).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- command line


def git_head(repository: Path) -> str:
    return subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def write_checks(check: Checks, path: Path) -> None:
    check.frame().to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sites", "adaredit", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--adaredit-root", type=Path, required=True,
                             help="a copy of the pinned AdarEdit reproducibility package (data/species/, "
                                  "data/species_prebalancing/, results/preds/)")
    sub.add_parser("score")
    sub.add_parser("gap-histogram")
    sub.choices["evaluate"].add_argument("--scores", type=Path, default=None,
                                         help="score table to evaluate (default: <output-dir>/scores.parquet)")
    sub.choices["evaluate"].add_argument("--adaredit-scores", type=Path, default=None,
                                         help="AdarEdit per-site scores (default: <output-dir>/adaredit_scores.parquet)")
    args = parser.parse_args()
    repository = args.repository.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repository / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    check = Checks()
    if args.command == "sites":
        sites = site_table(repository, args.adaredit_root, check)
        check.require("sites")
        sites.to_parquet(out / "sites.parquet", index=False)
        human = human_training_gaps(repository, HEADLINE_ENDPOINT)
        human.to_parquet(out / "human_artery_training_gaps.parquet", index=False)
        gap_summary(human, sites).to_csv(out / "gap_summary.csv", index=False)
        comparators(repository).to_csv(out / "comparators.csv", index=False)
        write_checks(check, out / "checks_sites.csv")
    elif args.command == "score":
        sites = pd.read_parquet(out / "sites.parquet")
        scores = score_sites(repository, sites, check)
        check.require("score")
        scores.to_parquet(out / "scores.parquet", index=False)
        write_checks(check, out / "checks_score.csv")
    elif args.command == "gap-histogram":
        human = pd.read_parquet(out / "human_artery_training_gaps.parquet")
        gap_histogram(human, pd.read_parquet(out / "sites.parquet")).to_csv(out / "gap_histogram.csv", index=False)
    elif args.command == "adaredit":
        sites = pd.read_parquet(out / "sites.parquet")
        adaredit_site_scores(args.adaredit_root, sites, check).to_parquet(out / "adaredit_scores.parquet", index=False)
        check.require("adaredit")
        write_checks(check, out / "checks_adaredit.csv")
    elif args.command == "evaluate":
        score_path = args.scores or out / "scores.parquet"
        scores = pd.read_parquet(score_path)
        scores = scores[scores.split == "test"]
        adaredit = pd.read_parquet(args.adaredit_scores or out / "adaredit_scores.parquet")
        D = scores.merge(adaredit, on=["species", "line_no"], validate="many_to_one")
        check("evaluate: every test row has AdarEdit per-site scores", len(scores), len(D),
              len(D) == len(scores) == len(CONTEXTS) * sum(v for (s, p), v in EXPECTED_ROWS.items() if p == "test"))
        comparator_table = comparators(repository)
        thresholds = {v: pd.read_csv(repository / f"results/matrix_{v}_threshold.csv", index_col=0)
                      for v in ADAREDIT_VARIANTS}
        record = {"what": "nonhuman transfer evaluation: AdarEdit's nonhuman test labels read against frozen scores",
                  "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
                  "repository_head": git_head(repository), "scores": {"path": score_path.name,
                                                                       "sha256": sha256_file(score_path)},
                  "inputs": {rel: sha256_file(args.adaredit_root / rel) for rel in
                             [f"data/species/{s}/test.jsonl" for s in ANIMALS]
                             + [f"data/species/{s}/test.metadata.csv" for s in ANIMALS]},
                  "bootstrap": {"replicates": NB, "seed": SEED, "unit": "genomic region"}}
        labels = test_labels(repository, args.adaredit_root, check)
        D = D.merge(labels, on=["species", "line_no"], validate="many_to_one")
        adaredit_anchor(repository, args.adaredit_root, labels, D, check)
        check.require("checks before any metric")
        M, C, F = analyse(D, comparator_table, thresholds, "label", check)
        check.require("evaluate")
        M.to_csv(out / "metrics.csv", index=False)
        C.to_csv(out / "contrasts.csv", index=False)
        F.to_csv(out / "f1.csv", index=False)
        D.to_parquet(out / "persite.parquet", index=False)
        record["outputs"] = {name: sha256_file(out / name) for name in ("metrics.csv", "contrasts.csv", "f1.csv")}
        (out / "run_record.json").write_text(json.dumps(record, indent=1) + "\n")
        write_checks(check, out / "checks_evaluate.csv")
    print(f"{len(check.rows)} checks, {sum(not r['pass'] for r in check.rows)} failed", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
