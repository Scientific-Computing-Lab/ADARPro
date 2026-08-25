"""Retrieve and validate GTEx v8 median expression for the prespecified RBP panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://gtexportal.org/api/v2"
TISSUE_MAP = {
    "Artery": "Artery_Tibial",
    "Brain": "Brain_Cerebellum",
    "Liver": "Liver",
    "MuscleSkeletal": "Muscle_Skeletal",
}
CONTROLS = ("ADARB1",)


def get_json(endpoint: str, parameters: dict) -> tuple[dict, str, bytes]:
    query = urllib.parse.urlencode(parameters, doseq=True)
    url = f"{API}/{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request) as response:
        body = response.read()
    return json.loads(body), url, body


def exact_gencode_id(gene: str) -> tuple[str, dict, str, bytes]:
    payload, url, body = get_json(
        "reference/geneSearch",
        {
            "geneId": gene,
            "gencodeVersion": "v26",
            "genomeBuild": "GRCh38/hg38",
            "itemsPerPage": 250,
        },
    )
    exact = [row for row in payload["data"] if row["geneSymbol"].upper() == gene.upper()]
    protein_coding = [row for row in exact if row.get("geneType") == "protein coding"]
    selected = protein_coding or exact
    if len(selected) > 1:
        hgnc_annotated = [
            row for row in selected if "Source:HGNC Symbol" in row.get("description", "")
        ]
        if len(hgnc_annotated) == 1:
            selected = hgnc_annotated
    if len(selected) != 1:
        raise RuntimeError(f"expected one exact GTEx v8 gene for {gene}, got {selected}")
    return selected[0]["gencodeId"], selected[0], url, body


def run(panel_path: Path, output_dir: Path) -> dict:
    panel = pd.read_csv(panel_path)
    genes = list(dict.fromkeys([*panel["gene"].tolist(), *CONTROLS]))
    retrieved = datetime.now(timezone.utc).isoformat()
    raw_dir = output_dir / "raw_api"
    raw_dir.mkdir(parents=True, exist_ok=True)
    references, provenance = [], []
    for gene in genes:
        gencode_id, reference, url, body = exact_gencode_id(gene)
        path = raw_dir / f"geneSearch_{gene}.json"
        path.write_bytes(body)
        references.append(
            {
                "gene": gene,
                "gencode_id": gencode_id,
                "gene_type": reference.get("geneType"),
                "chromosome": reference.get("chromosome"),
            }
        )
        provenance.append(
            {
                "kind": "gene_reference",
                "gene": gene,
                "url": url,
                "retrieved_utc": retrieved,
                "sha256": hashlib.sha256(body).hexdigest(),
                "local_path": str(path.resolve()),
            }
        )
    reference_table = pd.DataFrame(references)
    parameters = {
        "gencodeId": reference_table["gencode_id"].tolist(),
        "datasetId": "gtex_v8",
        "itemsPerPage": 100_000,
    }
    payload, url, body = get_json("expression/medianGeneExpression", parameters)
    expression_path = raw_dir / "medianGeneExpression.json"
    expression_path.write_bytes(body)
    provenance.append(
        {
            "kind": "median_expression",
            "gene": "ALL",
            "url": url,
            "retrieved_utc": retrieved,
            "sha256": hashlib.sha256(body).hexdigest(),
            "local_path": str(expression_path.resolve()),
        }
    )
    expression = pd.DataFrame(payload["data"]).rename(
        columns={"geneSymbol": "gene", "tissueSiteDetailId": "gtex_tissue", "median": "tpm"}
    )
    expected = set(genes)
    observed = set(expression["gene"])
    if observed != expected:
        raise RuntimeError(f"GTEx response gene mismatch: missing={expected-observed}, extra={observed-expected}")
    tissue_counts = expression.groupby("gene")["gtex_tissue"].nunique()
    if tissue_counts.nunique() != 1 or tissue_counts.iloc[0] != 54:
        raise RuntimeError(f"expected 54 GTEx v8 tissues per gene: {tissue_counts.to_dict()}")
    expression["log1p_tpm"] = np.log1p(expression["tpm"])
    expression["expression_percentile_across_gtex"] = expression.groupby("gene")[
        "tpm"
    ].rank(method="average", pct=True)
    context = expression[
        expression["gtex_tissue"].isin(TISSUE_MAP.values())
    ].copy()
    reverse_tissue_map = {value: key for key, value in TISSUE_MAP.items()}
    context["context"] = context["gtex_tissue"].map(reverse_tissue_map)
    context = context.merge(reference_table, on="gene", how="left", validate="many_to_one")
    if len(context) != len(genes) * len(TISSUE_MAP):
        raise RuntimeError("incomplete four-tissue GTEx expression matrix")
    adar = context[context["gene"] == "ADAR"][["context", "tpm"]].rename(
        columns={"tpm": "adar_tpm"}
    )
    adarb1 = context[context["gene"] == "ADARB1"][["context", "tpm"]].rename(
        columns={"tpm": "adarb1_tpm"}
    )
    context = context.merge(adar, on="context", validate="many_to_one").merge(
        adarb1, on="context", validate="many_to_one"
    )
    context["tpm_over_adar"] = context["tpm"] / context["adar_tpm"]
    context["tpm_over_adarb1"] = context["tpm"] / context["adarb1_tpm"]
    context["combined_context_permitted"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    expression.to_parquet(output_dir / "gtex_v8_all_tissues.parquet", index=False)
    context.to_csv(output_dir / "gtex_v8_context_expression.csv", index=False)
    reference_table.to_csv(output_dir / "gtex_v26_gene_reference.csv", index=False)
    pd.DataFrame(provenance).to_csv(output_dir / "provenance.csv", index=False)

    # Independent equivalence check against the upstream frozen ADAR/ADARB1 snapshot.
    upstream_path = output_dir.parents[2] / "analyses/gtex_tissue_selection/data/gtex_v8_adar_expression.csv"
    upstream_check = "not_available"
    if upstream_path.exists():
        upstream = pd.read_csv(upstream_path).rename(columns={"tissue": "gtex_tissue"})
        current = expression[expression["gene"].isin(["ADAR", "ADARB1"])].pivot(
            index="gtex_tissue", columns="gene", values="tpm"
        )
        merged = upstream.merge(current.reset_index(), on="gtex_tissue", validate="one_to_one")
        for gene in ("ADAR", "ADARB1"):
            if not np.allclose(merged[f"{gene}_median_TPM"], merged[gene], atol=0.011):
                raise RuntimeError(f"current GTEx API disagrees with upstream frozen {gene}")
        upstream_check = "54-tissue values agree after upstream two-decimal rounding"
    summary = {
        "dataset": "gtex_v8",
        "gencode_version": "v26",
        "genome_build": "GRCh38/hg38",
        "gene_count": len(genes),
        "tissue_count": int(expression["gtex_tissue"].nunique()),
        "individual_contexts": TISSUE_MAP,
        "combined_handling": "no expression vector assigned",
        "upstream_adar_snapshot_check": upstream_check,
        "retrieved_utc": retrieved,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--panel", type=Path, default=Path("context/rbp/candidate_panel.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/external/gtex_expression")
    )
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    print(json.dumps(run(resolve(args.panel), resolve(args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
