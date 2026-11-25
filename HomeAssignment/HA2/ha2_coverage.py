"""HA1-style C×S coverage on the revised HA2 inventory; no perturbation."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import ha2_analysis as analysis
import sync as input_sync


ROOT = Path(__file__).resolve().parent
FEATURES = [
    "visual", "auditory", "tactile", "motor", "proprioceptive", "interoceptive", "social"
]
QUADRANTS = ["C+ S+", "C+ S-", "C- S+", "C- S-"]
THRESHOLD = 0.5
MIN_PER_QUADRANT = 3


def minmax(values):
    span = values.max() - values.min()
    if pd.isna(span) or span <= 0:
        raise ValueError("Min-max normalization requires variation")
    return (values - values.min()) / span


def quadrant_coverage(scores, concepts=None, *, data_dir=ROOT):
    """C=sum(7 features); S=shortest baseline depth; normalize over targets.

    Keep the prior role 4 policy: an isolated node has no usable S here.
    This does NOT reuse HA2's normalized S_depth or include auxiliary nodes.
    """
    if concepts is None:
        concepts = pd.read_csv(
            Path(data_dir) / "raw/Concepts_hw2.csv",
            dtype={"concept_id": str}, keep_default_na=False,
        )
    required_concepts = {"concept_id", "term", *FEATURES}
    required_scores = {"node_id", "label", "is_target", "depth_min", "taxonomy_isolate"}
    if not required_concepts <= set(concepts) or not required_scores <= set(scores):
        raise ValueError("Missing coverage input columns")
    if not concepts.concept_id.is_unique or not scores.node_id.is_unique:
        raise ValueError("Duplicate concept or score IDs")
    target_flags = scores.is_target.map({"TRUE": True, "FALSE": False, True: True, False: False})
    if target_flags.isna().any():
        raise ValueError("Invalid is_target in baseline scores")
    targets = scores.loc[target_flags].set_index("node_id")
    concepts = concepts.set_index("concept_id")
    if (set(concepts.index) != set(targets.index)
            or not concepts.term.sort_index().equals(targets.label.sort_index().rename("term"))):
        raise ValueError("Target IDs or labels differ between coverage inputs")
    features = concepts[FEATURES].apply(pd.to_numeric)
    if not features.isin([0, 1, 2]).all().all():
        raise ValueError("Unexpected feature scale or missing value")
    isolates = targets.taxonomy_isolate.map({True: True, False: False, "True": True, "False": False})
    if isolates.isna().any():
        raise ValueError("Invalid taxonomy_isolate in baseline scores")
    depths = pd.to_numeric(targets.depth_min)
    if depths.isna().any() or not depths.ge(0).all() or not depths.mod(1).eq(0).all():
        raise ValueError("Invalid raw taxonomic depth")
    coverage = features.astype(int).copy()
    coverage.insert(0, "term", concepts.term)
    coverage["C_raw"] = features.sum(axis=1).astype(int)
    coverage["S_raw"] = depths.mask(isolates)
    coverage["C_norm"] = minmax(coverage.C_raw)
    coverage["S_norm"] = minmax(coverage.S_raw)
    coverage["quadrant"] = (
        coverage.C_norm.ge(THRESHOLD).map({True: "C+", False: "C-"})
        + " " + coverage.S_norm.ge(THRESHOLD).map({True: "S+", False: "S-"})
    ).where(coverage.S_raw.notna(), "no S")
    coverage["S_missing_reason"] = isolates.map({True: "isolated_in_baseline", False: ""})
    return coverage.rename_axis("node_id").sort_index()


def counts(coverage):
    return coverage.quadrant.value_counts().reindex(QUADRANTS + ["no S"], fill_value=0)


def normalization(coverage):
    return {
        "target_count": len(coverage), "usable_S_count": int(coverage.S_raw.notna().sum()),
        "C_min": int(coverage.C_raw.min()), "C_max": int(coverage.C_raw.max()),
        "S_min": float(coverage.S_raw.min()), "S_max": float(coverage.S_raw.max()),
        "C_threshold": THRESHOLD, "S_threshold": THRESHOLD,
        "minimum_per_quadrant": MIN_PER_QUADRANT,
    }


def summary(coverage):
    sizes = counts(coverage)
    return {
        **normalization(coverage),
        "quadrant_counts": {q: int(sizes[q]) for q in QUADRANTS},
        "no_S_count": int(sizes["no S"]),
        "quadrant_minimum_met": bool(sizes.loc[QUADRANTS].ge(MIN_PER_QUADRANT).all()),
        "all_targets_have_S": bool(coverage.S_raw.notna().all()),
        "on_C_threshold": coverage.loc[coverage.C_norm.eq(THRESHOLD), "term"].tolist(),
        "on_S_threshold": coverage.loc[coverage.S_norm.eq(THRESHOLD), "term"].tolist(),
    }


def baseline_identity(manifest):
    # Status flags may change as later stages finish. Numerical lineage may not.
    identity = {key: manifest[key] for key in (
        "source_commit", "input_sha256", "code_sha256",
        "baseline_output_sha256", "primary_normalization",
    )}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def checked_baseline(data_dir):
    data_dir = Path(data_dir)
    analysis.load_inputs(data_dir)  # Check revised input schema/identities/POS.
    manifest = json.loads((data_dir / "results/run_manifest.json").read_text())
    if manifest.get("run_kind") != "baseline":
        raise ValueError("Expected a baseline run manifest")
    for group, folder in (
        ("input_sha256", data_dir),
        ("baseline_output_sha256", data_dir / "results"),
    ):
        for name, expected in manifest[group].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Stale baseline dependency: {name}; recalculate baseline first")
    code_paths = {
        "ha2_analysis.py": Path(analysis.__file__), "sync.py": Path(input_sync.__file__),
    }
    for name, expected in manifest["code_sha256"].items():
        if hashlib.sha256(code_paths[name].read_bytes()).hexdigest() != expected:
            raise ValueError(f"Baseline code changed: {name}; recalculate baseline first")
    scores = pd.read_csv(
        data_dir / "results/specificity.csv",
        dtype={"node_id": str, "is_target": str}, keep_default_na=False,
    )
    return scores, manifest


def run_coverage(data_dir=ROOT):
    data_dir = Path(data_dir)
    scores, baseline = checked_baseline(data_dir)
    concepts = pd.read_csv(
        data_dir / "raw/Concepts_hw2.csv", dtype={"concept_id": str}, keep_default_na=False
    )
    primary = quadrant_coverage(scores, concepts)
    conservative_concepts = concepts.copy(deep=True)
    selected = conservative_concepts.term.eq("близость")
    if selected.sum() != 1 or not conservative_concepts.loc[selected, "social"].eq(1).all():
        raise ValueError("Expected one provisional social=1 rating for «близость»")
    conservative_concepts.loc[selected, "social"] = 0
    conservative = quadrant_coverage(scores, conservative_concepts)
    comparison = primary[["term", "C_raw", "C_norm", "S_raw", "S_norm", "quadrant"]].add_suffix(
        "_as_provided"
    ).join(conservative[["C_raw", "C_norm", "S_raw", "S_norm", "quadrant"]].add_suffix("_social0"))
    comparison["quadrant_changed"] = comparison.quadrant_as_provided.ne(comparison.quadrant_social0)
    norms = pd.DataFrame([
        {"scenario": name, **normalization(table)}
        for name, table in (("as_provided", primary), ("closeness_social_0", conservative))
    ])
    feature_rows = []
    for name, table in (("as_provided", primary), ("closeness_social_0", conservative)):
        for feature in FEATURES:
            feature_rows.append({
                "scenario": name, "feature": feature,
                **{f"count_{value}": int(table[feature].eq(value).sum()) for value in (0, 1, 2)},
                "nonzero_count": int(table[feature].gt(0).sum()),
                "sum": int(table[feature].sum()),
            })
    report = {
        "scope": "HA1-style C×S coverage of the revised HA2 target inventory, not a full HA1 audit",
        "source_commit": baseline["source_commit"],
        "as_provided": summary(primary),
        "closeness_social_0": summary(conservative),
        "sensitivity_changed_concepts": comparison.loc[
            comparison.quadrant_changed, "term_as_provided"
        ].tolist(),
        "source_annotations_modified": False, "perturbation_performed": False,
    }
    output = data_dir / "results/ha1_coverage"
    output.mkdir(exist_ok=True)
    for name, table in (
        ("quadrant_coverage.csv", primary), ("quadrant_counts.csv", counts(primary).to_frame("count")),
        ("quadrant_coverage_social0.csv", conservative),
        ("quadrant_counts_social0.csv", counts(conservative).to_frame("count")),
        ("sensitivity_comparison.csv", comparison),
    ):
        table.to_csv(output / name, float_format="%.15g", lineterminator="\n")
    norms.to_csv(output / "normalization.csv", index=False, float_format="%.15g", lineterminator="\n")
    pd.DataFrame(feature_rows).to_csv(output / "feature_coverage.csv", index=False, lineterminator="\n")
    (output / "requirements.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_names = [
        "quadrant_coverage.csv", "quadrant_counts.csv", "quadrant_coverage_social0.csv",
        "quadrant_counts_social0.csv", "sensitivity_comparison.csv",
        "normalization.csv", "feature_coverage.csv", "requirements.json",
    ]
    reference = data_dir.parent / "HA1_submission/HA1_C_S_analysis.ipynb"
    manifest = {
        "run_kind": "ha1_coverage",
        "source_commit": baseline["source_commit"],
        "baseline_identity_sha256": baseline_identity(baseline),
        "baseline_output_sha256": baseline["baseline_output_sha256"],
        "coverage_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {
            name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in ("raw/Concepts_hw2.csv", "nodes.csv", "edges.csv", "results/specificity.csv")
        },
        "method": {
            "C": "sum of seven source ratings; each in {0,1,2}",
            "S": "baseline depth_min; isolated nodes masked",
            "normalization": "separate min-max over targets; missing S excluded",
            "threshold": THRESHOLD, "minimum_per_quadrant": MIN_PER_QUADRANT,
            "equality_at_threshold": "plus",
        },
        "HA1_method_reference": {
            "path": "HomeAssignment/HA1_submission/HA1_C_S_analysis.ipynb",
            "cells_zero_based": [1, 6, 8, 11, 12],
            "sha256": hashlib.sha256(reference.read_bytes()).hexdigest() if reference.exists() else None,
        },
        "output_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in output_names
        },
        "perturbation_performed": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    baseline["coverage_status"] = "computed"
    baseline["coverage_manifest"] = "ha1_coverage/run_manifest.json"
    baseline["artifacts_status"] = "not_recomputed"
    baseline.pop("artifacts_manifest", None)
    (data_dir / "results/run_manifest.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run_coverage(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
