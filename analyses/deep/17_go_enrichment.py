#!/usr/bin/env python
"""GO enrichment of PT-specific genes and co-degradation modules.

Uses gprofiler-official for functional annotation (no internet needed
if cached; falls back to simple keyword matching on gene names).
"""
from _common import *

OUT = output_dir("17_go_enrichment")


def run_gprofiler(gene_list, organism="mmusculus"):
    """Run g:Profiler enrichment. Returns DataFrame or None."""
    try:
        from gprofiler import GProfiler
        gp = GProfiler(return_dataframe=True)
        result = gp.profile(organism=organism, query=gene_list)
        return result
    except ImportError:
        print("  [WARN] gprofiler-official not installed. Using fallback.")
        return None
    except Exception as e:
        print(f"  [WARN] g:Profiler failed: {e}")
        return None


def simple_gene_annotation(gene_list):
    """Fallback: annotate genes with known function keywords."""
    # Known RNA-binding / degradation related genes
    rbp_keywords = {
        "Igf2bp": "RNA binding protein", "Hnrnp": "RNA binding protein",
        "Rbfox": "RNA binding protein", "Elavl": "RNA binding protein",
        "Srsf": "splicing factor", "Mbnl": "splicing factor",
        "Cnot": "deadenylase complex", "Pan3": "deadenylase",
        "Snd1": "RNA binding", "Fus": "RNA binding protein",
        "Nrxn": "neuronal adhesion", "Kcnma": "ion channel",
        "Rora": "transcription factor", "Rfx": "transcription factor",
        "Ptprn": "protein tyrosine phosphatase",
        "Trim": "E3 ubiquitin ligase",
    }

    annotations = {}
    for gene in gene_list:
        for kw, ann in rbp_keywords.items():
            if kw.lower() in gene.lower():
                annotations[gene] = ann
                break
    return annotations


def main():
    set_figure_style()

    all_results = {}

    for name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{name.upper()}: GO enrichment\n{'=' * 60}")

        # Load PT-specific genes
        adv_file = PROJECT_ROOT / "output" / "deep_advantages" / "results" / f"{name}_advantages.json"
        if not adv_file.exists():
            print("  [SKIP] No advantage results")
            continue

        with open(adv_file) as f:
            adv = json.load(f)

        pt_genes = adv.get("disentanglement", {}).get("pt_specific_genes", [])
        if not pt_genes:
            print("  No PT-specific genes")
            continue

        print(f"  PT-specific genes: {len(pt_genes)}")

        # Try g:Profiler
        organism = "mmusculus" if name in ("pancreas", "dentate_gyrus") else "hsapiens"
        go_result = run_gprofiler(pt_genes, organism=organism)

        ds_results = {"pt_genes": pt_genes, "organism": organism}

        if go_result is not None and len(go_result) > 0:
            # Filter significant results
            sig = go_result[go_result["p_value"] < 0.05].sort_values("p_value")
            top_terms = sig.head(20)[["source", "native", "name", "p_value", "intersection_size"]].to_dict("records")
            print(f"  g:Profiler: {len(sig)} significant terms")
            for t in top_terms[:10]:
                print(f"    {t['source']}:{t['name']} (p={t['p_value']:.2e}, n={t['intersection_size']})")
            ds_results["go_terms"] = top_terms
            ds_results["n_significant"] = len(sig)
        else:
            # Fallback
            annotations = simple_gene_annotation(pt_genes)
            print(f"  Fallback annotations: {len(annotations)}/{len(pt_genes)} annotated")
            for gene, ann in sorted(annotations.items()):
                print(f"    {gene}: {ann}")
            ds_results["fallback_annotations"] = annotations

        # Load co-degradation modules
        coexpr_file = PROJECT_ROOT / "output" / "deep_benchmarks" / "09_gamma_coexpression" / "results" / f"{name}_gamma_coexpression.json"
        if coexpr_file.exists():
            with open(coexpr_file) as f:
                coexpr = json.load(f)

            print(f"\n  Co-degradation modules:")
            module_go = []
            for mod in coexpr.get("modules", []):
                mod_genes = mod.get("example_genes", [])
                if len(mod_genes) < 5:
                    continue

                go_mod = run_gprofiler(mod_genes, organism=organism)
                if go_mod is not None and len(go_mod) > 0:
                    top = go_mod[go_mod["p_value"] < 0.05].head(3)
                    terms = top["name"].tolist() if len(top) > 0 else []
                else:
                    terms = list(simple_gene_annotation(mod_genes).values())[:3]

                module_go.append({
                    "module": mod["module"],
                    "n_genes": mod["n_genes"],
                    "top_terms": terms,
                    "top_rbps": mod.get("top_rbps", []),
                })
                if terms:
                    print(f"    Module {mod['module']} ({mod['n_genes']} genes): {', '.join(terms[:3])}")

            ds_results["module_go"] = module_go

        all_results[name] = ds_results

    save_json(all_results, "go_enrichment", OUT)

    # Summary figure: enrichment barplot for top terms
    for name, ds in all_results.items():
        terms = ds.get("go_terms", [])
        if not terms:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        term_names = [t["name"][:40] for t in terms[:10]]
        pvals = [-np.log10(t["p_value"]) for t in terms[:10]]
        ax.barh(term_names[::-1], pvals[::-1], color="darkorange", alpha=0.7)
        ax.set_xlabel("-log10(p-value)")
        ax.set_title(f"{name}: GO enrichment of PT-specific genes")
        fig.tight_layout()
        save_fig(fig, f"{name}_go_enrichment", OUT)


if __name__ == "__main__":
    main()
