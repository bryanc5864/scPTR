#!/usr/bin/env python3
"""
Download ENCODE eCLIP data for specific RBPs and extract gene-level targets.

Steps:
1. Query ENCODE REST API for eCLIP experiments per RBP
2. Download IDR-merged peak files (GRCh38, bed narrowPeak)
3. Map peak coordinates to genes using Ensembl gene annotations (via pybiomart)
4. Save result as CSV: rbp, target_gene

For RBPs without direct ENCODE eCLIP data, close family members are used
as proxies where available (e.g., TRA2A for TRA2B, MBNL1 for MBNL2).

Author: auto-generated
"""

import os
import sys
import time
import gzip
import bisect
import logging
from collections import defaultdict

import requests
import pandas as pd
from pybiomart import Server

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Primary RBPs requested by the user
RBPS = [
    "HNRNPA1", "YBX1", "ELAVL1", "SRSF3", "RBFOX2", "FUS",
    "HNRNPD", "CELF2", "MATR3", "HNRNPC", "HNRNPU", "RBFOX3",
    "ELAVL3", "ELAVL4", "ZFP36L1", "TRA2B", "MBNL2",
]

# Mapping from requested RBP -> ENCODE target label.
# Where the exact gene is not in ENCODE eCLIP, we use a closely related
# family member as a proxy. These are noted in ENCODE_LABEL_NOTE.
ENCODE_LABEL = {
    "HNRNPA1": "HNRNPA1",
    "YBX1": None,           # Not in ENCODE eCLIP (YBX3 is, but too divergent)
    "ELAVL1": "ELAVL1",
    "SRSF3": None,           # Not in ENCODE eCLIP
    "RBFOX2": "RBFOX2",
    "FUS": "FUS",
    "HNRNPD": None,           # Not in ENCODE eCLIP
    "CELF2": None,           # Not in ENCODE eCLIP
    "MATR3": "MATR3",
    "HNRNPC": "HNRNPC",
    "HNRNPU": "HNRNPU",
    "RBFOX3": None,           # Not in ENCODE eCLIP
    "ELAVL3": None,           # Not in ENCODE eCLIP (ELAVL1 too divergent in targets)
    "ELAVL4": None,           # Not in ENCODE eCLIP
    "ZFP36L1": None,           # Not in ENCODE eCLIP
    "TRA2B": "TRA2A",        # TRA2A is close paralog, same eCLIP binding profile
    "MBNL2": "MBNL1",        # MBNL1 is close paralog
}

# Notes about proxy usage
ENCODE_LABEL_NOTE = {
    "TRA2B": "proxy:TRA2A",
    "MBNL2": "proxy:MBNL1",
}

OUTPUT_CSV = "/home/bcheng/scPTR/src/scptr/benchmark/data/eclip_targets.csv"

ENCODE_BASE = "https://www.encodeproject.org"
ENCODE_HEADERS = {"Accept": "application/json"}

# Sleep between ENCODE API requests (seconds)
REQUEST_DELAY = 0.3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Build gene coordinate index from Ensembl (GRCh38)
# ---------------------------------------------------------------------------

def fetch_gene_annotations() -> pd.DataFrame:
    """
    Retrieve gene annotations from Ensembl BioMart (GRCh38).
    Keeps protein-coding genes and lncRNAs on standard chromosomes.
    Returns a DataFrame with columns: chrom, start, end, strand, gene_name
    """
    log.info("Fetching gene annotations from Ensembl BioMart ...")
    server = Server(host="http://www.ensembl.org")
    dataset = server["ENSEMBL_MART_ENSEMBL"]["hsapiens_gene_ensembl"]

    result = dataset.query(
        attributes=[
            "chromosome_name",
            "start_position",
            "end_position",
            "strand",
            "external_gene_name",
            "gene_biotype",
        ],
    )

    result.columns = ["chrom", "start", "end", "strand", "gene_name", "biotype"]

    # Keep protein-coding genes and lncRNAs (commonly bound by RBPs)
    keep_biotypes = {"protein_coding", "lncRNA"}
    result = result[result["biotype"].isin(keep_biotypes)].copy()

    # Only keep standard chromosomes (1-22, X, Y)
    standard_chroms = {str(c) for c in range(1, 23)} | {"X", "Y"}
    result = result[result["chrom"].isin(standard_chroms)].copy()

    # Add 'chr' prefix to match ENCODE bed files
    result["chrom"] = "chr" + result["chrom"].astype(str)

    # Drop rows without gene names
    result = result[result["gene_name"].notna() & (result["gene_name"] != "")].copy()
    result = result.drop(columns=["biotype"]).reset_index(drop=True)

    log.info(f"  Retrieved {len(result):,} gene annotations")
    return result


def build_gene_index(genes_df: pd.DataFrame) -> dict:
    """
    Build a chromosome-indexed dict for fast overlap queries.
    Returns: {chrom: list of (start, end, gene_name)} sorted by start.
    """
    index = defaultdict(list)
    for _, row in genes_df.iterrows():
        index[row["chrom"]].append(
            (int(row["start"]), int(row["end"]), row["gene_name"])
        )

    # Sort each chromosome by start position
    for chrom in index:
        index[chrom].sort(key=lambda x: x[0])

    return dict(index)


def find_overlapping_genes(
    chrom: str, peak_start: int, peak_end: int, gene_index: dict
) -> set:
    """
    Find all genes whose genomic interval overlaps with a peak region.
    Uses binary search on sorted gene starts for efficiency.
    """
    genes = gene_index.get(chrom, [])
    if not genes:
        return set()

    starts = [g[0] for g in genes]
    # Find the index of the first gene whose start >= peak_end
    right_idx = bisect.bisect_left(starts, peak_end)

    overlapping = set()

    # Scan backwards from right_idx to find all genes overlapping the peak.
    # A gene overlaps if gene_start < peak_end AND gene_end > peak_start.
    # Since genes are sorted by start, once gene_start drops well below
    # peak_start we use a distance cutoff to stop (genes can be long).
    for i in range(max(0, right_idx - 1), -1, -1):
        g_start, g_end, g_name = genes[i]
        if g_start < peak_end and g_end > peak_start:
            overlapping.add(g_name)
        # Safety cutoff: stop if gene starts > 2 Mb before peak start
        if g_start < peak_start - 2_000_000:
            break

    # Also check a few genes forward (edge cases at the boundary)
    for i in range(right_idx, min(len(genes), right_idx + 10)):
        g_start, g_end, g_name = genes[i]
        if g_start >= peak_end:
            break
        if g_start < peak_end and g_end > peak_start:
            overlapping.add(g_name)

    return overlapping


# ---------------------------------------------------------------------------
# Step 2: Query ENCODE API for eCLIP experiments
# ---------------------------------------------------------------------------

def search_eclip_experiments(target_label: str) -> list[dict]:
    """
    Search ENCODE for eCLIP experiments targeting a given gene label (human).
    Returns list of experiment info dicts.
    """
    url = f"{ENCODE_BASE}/search/"
    params = {
        "type": "Experiment",
        "assay_title": "eCLIP",
        "target.label": target_label,
        "status": "released",
        "format": "json",
        "limit": "all",
    }

    try:
        r = requests.get(url, params=params, headers=ENCODE_HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"  API search failed for {target_label}: {e}")
        return []

    data = r.json()
    experiments = []
    for exp in data.get("@graph", []):
        experiments.append({
            "accession": exp["accession"],
            "biosample_summary": exp.get("biosample_summary", "unknown"),
            "target_label": exp.get("target", {}).get("label", target_label),
        })

    return experiments


def find_idr_peaks_file(experiment_accession: str) -> dict | None:
    """
    For an experiment, find the IDR-merged peaks file (GRCh38, bed narrowPeak).
    The IDR-merged file has biological_replicates containing both rep1 and rep2.
    Falls back to any released bed narrowPeak if IDR-merged not found.
    """
    url = (
        f"{ENCODE_BASE}/experiments/{experiment_accession}/"
        f"?format=json&frame=embedded"
    )

    try:
        r = requests.get(url, headers=ENCODE_HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"  Failed to get experiment {experiment_accession}: {e}")
        return None

    exp_data = r.json()
    files = exp_data.get("files", [])

    # Find bed narrowPeak files that are released, preferring IDR-merged
    # (biological_replicates has 2+ entries)
    idr_candidates = []
    single_rep_candidates = []

    for f in files:
        if not isinstance(f, dict):
            continue
        if (
            f.get("output_type") == "peaks"
            and f.get("file_format") == "bed"
            and f.get("file_format_type") == "narrowPeak"
            and f.get("status") == "released"
        ):
            if len(f.get("biological_replicates", [])) >= 2:
                idr_candidates.append(f)
            else:
                single_rep_candidates.append(f)

    candidates = idr_candidates if idr_candidates else single_rep_candidates
    if not candidates:
        return None

    # Prefer GRCh38 assembly
    for f in candidates:
        if f.get("assembly") == "GRCh38":
            return {
                "accession": f.get("accession"),
                "href": f.get("href"),
                "assembly": f.get("assembly"),
                "biological_replicates": f.get("biological_replicates"),
            }

    # Fall back to any assembly
    f = candidates[0]
    return {
        "accession": f.get("accession"),
        "href": f.get("href"),
        "assembly": f.get("assembly"),
        "biological_replicates": f.get("biological_replicates"),
    }


# ---------------------------------------------------------------------------
# Step 3: Download and parse peak files
# ---------------------------------------------------------------------------

def download_and_parse_peaks(href: str) -> list[tuple]:
    """
    Download a bed.gz file from ENCODE and parse it.
    Returns list of (chrom, start, end) tuples.
    """
    url = ENCODE_BASE + href

    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"  Failed to download {url}: {e}")
        return []

    content = r.content
    try:
        text = gzip.decompress(content).decode("utf-8", errors="replace")
    except gzip.BadGzipFile:
        text = content.decode("utf-8", errors="replace")

    peaks = []
    for line in text.strip().split("\n"):
        if not line or line.startswith("#") or line.startswith("track"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        chrom = fields[0]
        try:
            start = int(fields[1])
            end = int(fields[2])
        except ValueError:
            continue
        peaks.append((chrom, start, end))

    return peaks


# ---------------------------------------------------------------------------
# Step 4: Map peaks to genes
# ---------------------------------------------------------------------------

def map_peaks_to_genes(peaks: list[tuple], gene_index: dict) -> set:
    """Map a list of peaks to overlapping gene names."""
    all_genes = set()
    for chrom, start, end in peaks:
        genes = find_overlapping_genes(chrom, start, end, gene_index)
        all_genes.update(genes)
    return all_genes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 1: Get gene annotations
    genes_df = fetch_gene_annotations()
    gene_index = build_gene_index(genes_df)
    log.info(f"Gene index built for {len(gene_index)} chromosomes")

    # Step 2-4: For each RBP, query ENCODE, download peaks, map to genes
    all_results = []  # list of (rbp, target_gene) tuples
    rbps_found = []
    rbps_not_found = []

    for rbp in RBPS:
        encode_label = ENCODE_LABEL.get(rbp, rbp)
        if encode_label is None:
            log.warning(f"Skipping {rbp} -- no ENCODE eCLIP data available")
            rbps_not_found.append(rbp)
            continue

        note = ENCODE_LABEL_NOTE.get(rbp, "")
        if note:
            log.info(f"Processing {rbp} (using {note}) ...")
        else:
            log.info(f"Processing {rbp} ...")

        time.sleep(REQUEST_DELAY)

        # Search for experiments
        experiments = search_eclip_experiments(encode_label)
        if not experiments:
            log.warning(f"  No eCLIP experiments found for {encode_label}")
            rbps_not_found.append(rbp)
            continue

        log.info(f"  Found {len(experiments)} experiment(s)")

        rbp_targets = set()

        for exp in experiments:
            acc = exp["accession"]
            biosample = exp["biosample_summary"]
            log.info(f"  Experiment {acc} ({biosample})")
            time.sleep(REQUEST_DELAY)

            # Find IDR peaks file
            peaks_file = find_idr_peaks_file(acc)
            if not peaks_file:
                log.warning(f"    No peaks file found for {acc}")
                continue

            log.info(
                f"    Peaks file: {peaks_file['accession']} "
                f"(assembly={peaks_file['assembly']}, "
                f"bio_reps={peaks_file['biological_replicates']})"
            )
            time.sleep(REQUEST_DELAY)

            # Download and parse peaks
            peaks = download_and_parse_peaks(peaks_file["href"])
            if not peaks:
                log.warning(
                    f"    No peaks parsed from {peaks_file['accession']}"
                )
                continue

            log.info(f"    Downloaded {len(peaks):,} peaks")

            # Map peaks to genes
            target_genes = map_peaks_to_genes(peaks, gene_index)
            log.info(f"    Mapped to {len(target_genes):,} unique target genes")

            rbp_targets.update(target_genes)

        if rbp_targets:
            rbps_found.append(rbp)
            for gene in sorted(rbp_targets):
                all_results.append((rbp, gene))
            log.info(
                f"  Total unique targets for {rbp}: {len(rbp_targets):,}"
            )
        else:
            rbps_not_found.append(rbp)
            log.warning(f"  No targets found for {rbp}")

    # Step 5: Save results
    if all_results:
        df = pd.DataFrame(all_results, columns=["rbp", "target_gene"])
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        log.info(f"\nSaved {len(df):,} RBP-target pairs to {OUTPUT_CSV}")
        log.info(f"RBPs with data: {sorted(rbps_found)}")
        log.info(f"RBPs without data: {sorted(rbps_not_found)}")
        log.info("\nSummary per RBP:")
        for rbp, group in df.groupby("rbp"):
            note = ENCODE_LABEL_NOTE.get(rbp, "")
            suffix = f" ({note})" if note else ""
            log.info(f"  {rbp}: {len(group):,} target genes{suffix}")
    else:
        log.error("No results found for any RBP!")
        sys.exit(1)


if __name__ == "__main__":
    main()
