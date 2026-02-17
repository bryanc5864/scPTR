"""Tests for prior-weighted network inference."""

import numpy as np
import pandas as pd
import pytest


def test_network_with_priors(analyzed_adata):
    """Network inference with priors runs and returns DataFrame."""
    import scptr

    gene_names = analyzed_adata.var_names.tolist()
    regulators = gene_names[:10]
    targets = gene_names[10:15]

    # Create a prior network
    prior_rows = []
    for reg in regulators[:3]:
        for tgt in targets:
            prior_rows.append({"regulator": reg, "target": tgt, "weight": 2.0})
    prior_network = pd.DataFrame(prior_rows)

    result = scptr.tl.infer_network(
        analyzed_adata,
        regulators=regulators,
        targets=targets,
        prior_network=prior_network,
    )

    assert isinstance(result, pd.DataFrame)
    if len(result) > 0:
        assert set(result.columns) == {"regulator", "target", "weight"}


def test_network_without_priors_unchanged(analyzed_adata):
    """Network inference without priors still works as before."""
    import scptr

    gene_names = analyzed_adata.var_names.tolist()
    result = scptr.tl.infer_network(
        analyzed_adata,
        regulators=gene_names[:10],
        targets=gene_names[10:15],
    )

    assert isinstance(result, pd.DataFrame)
    params = analyzed_adata.uns["scptr"]["infer_network"]
    assert params["has_prior"] is False


def test_network_prior_logs_has_prior(analyzed_adata):
    """Prior usage is logged in uns params."""
    import scptr

    gene_names = analyzed_adata.var_names.tolist()
    prior_network = pd.DataFrame(
        {"regulator": [gene_names[0]], "target": [gene_names[10]], "weight": [1.0]}
    )

    scptr.tl.infer_network(
        analyzed_adata,
        regulators=gene_names[:5],
        targets=gene_names[10:12],
        prior_network=prior_network,
    )

    params = analyzed_adata.uns["scptr"]["infer_network"]
    assert params["has_prior"] is True


def test_load_motif_priors_validates_columns(tmp_path):
    """load_motif_priors raises on missing columns."""
    import scptr

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("col_a,col_b\n1,2\n")

    with pytest.raises(ValueError, match="missing required columns"):
        scptr.tl.load_motif_priors(str(bad_csv))


def test_load_motif_priors_valid(tmp_path):
    """load_motif_priors loads a valid CSV."""
    import scptr

    csv_path = tmp_path / "priors.csv"
    csv_path.write_text("regulator,target,weight\nA,B,1.5\nC,D,0.5\n")

    df = scptr.tl.load_motif_priors(str(csv_path))
    assert len(df) == 2
    assert list(df.columns) == ["regulator", "target", "weight"]


def test_list_known_rbps():
    """list_known_rbps returns a non-empty list."""
    import scptr

    rbps = scptr.tl.list_known_rbps()
    assert len(rbps) > 100

    human_rbps = scptr.tl.list_known_rbps(organism="human")
    mouse_rbps = scptr.tl.list_known_rbps(organism="mouse")
    assert len(human_rbps) > 0
    assert len(mouse_rbps) > 0
    assert len(human_rbps) + len(mouse_rbps) == len(rbps)
