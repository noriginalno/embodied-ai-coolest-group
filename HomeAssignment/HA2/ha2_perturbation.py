"""Recalculate HA1 coverage and run the HA2 edge perturbation experiment."""
from pathlib import Path
import json

import networkx as nx
import pandas as pd

from ha2_analysis import build_graphs, compute_specificity, load_inputs, validate_scores

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/role4"
FEATURES = [
    "visual", "auditory", "tactile", "motor", "proprioceptive", "interoceptive", "social"
]
ADDITIONS = [
    ("A07", "is_a", "A08", "manual", "medium",
     "Линейный размер является величиной."),
    ("A11", "is_a", "C05", "manual", "medium",
     "Высота относительно уровня отсчёта является количественным параметром положения."),
    ("C10", "associated_with", "C11", "manual", "high",
     "Ширина и глубина используются как пространственные измерения объекта."),
]
REMOVALS = [
    ("C01", "is_a", "C18"),
    ("A05", "is_a", "C13"),
    ("A11", "is_a", "C05"),
]
KEYS = ["source", "relation", "target"]
METRICS = ["S_depth", "S_breadth"]
STATS = ["depth_min", "depth_max", "descendants", "parent_count", "root_count",
         "taxonomy_isolate", *METRICS]


def minmax(values):
    span = values.max() - values.min()
    if span <= 0:
        raise ValueError("Min-max normalization requires variation")
    return (values - values.min()) / span


def quadrant_coverage(scores):
    concepts = pd.read_csv(ROOT / "concepts.csv").set_index("concept_id")
    targets = scores[scores.is_target == "TRUE"].set_index("node_id")
    if len(concepts) != 25 or not concepts.index.is_unique:
        raise ValueError("Expected 25 unique concepts")
    if not concepts.term.sort_index().equals(targets.label.sort_index().rename("term")):
        raise ValueError("Target IDs or labels differ between inputs")
    features = concepts[FEATURES].apply(pd.to_numeric)
    if not features.isin([0, 1, 2]).all().all():
        raise ValueError("Unexpected feature scale or missing value")
    coverage = pd.DataFrame({"term": concepts.term, "C_raw": features.sum(axis=1)})
    coverage["S_raw"] = targets.depth_min.mask(targets.taxonomy_isolate)
    coverage["C_norm"] = minmax(coverage.C_raw)
    coverage["S_norm"] = minmax(coverage.S_raw)
    coverage["quadrant"] = (
        coverage.C_norm.ge(0.5).map({True: "C+", False: "C-"})
        + " " + coverage.S_norm.ge(0.5).map({True: "S+", False: "S-"})
    ).where(coverage.S_raw.notna(), "no S")
    return coverage


def evaluate(nodes, edges):
    full, taxonomy = build_graphs(nodes, edges)
    scores, constants = compute_specificity(taxonomy)
    validate_scores(taxonomy, scores)
    return full, taxonomy, scores, constants


def main():
    nodes, edges = load_inputs(ROOT)
    full, taxonomy, baseline, constants = evaluate(nodes, edges)
    reference = pd.read_csv(ROOT / "results/specificity.csv")
    pd.testing.assert_frame_equal(
        baseline.set_index("node_id")[STATS].sort_index(),
        reference.set_index("node_id")[STATS].sort_index(),
        check_dtype=False,
    )
    coverage = quadrant_coverage(baseline)
    print(coverage.quadrant.value_counts().to_string())

    additions = pd.DataFrame(ADDITIONS, columns=edges.columns)
    original_keys = pd.MultiIndex.from_frame(edges[KEYS])
    assert not pd.MultiIndex.from_frame(additions[KEYS]).isin(original_keys).any()
    assert taxonomy.in_degree("A08") == 0 and taxonomy.in_degree("C05") == 1
    assert taxonomy.out_degree("C18") == 0
    assert taxonomy.in_degree("A05") > 0 and taxonomy.out_degree("C13") > 0

    added = pd.concat([edges, additions], ignore_index=True)
    _, after_add_graph, after_add, add_constants = evaluate(nodes, added)
    _, _, tax_only, _ = evaluate(nodes, added[added.relation == "is_a"])
    pd.testing.assert_frame_equal(after_add, tax_only)

    removal_mask = pd.MultiIndex.from_frame(added[KEYS]).isin(REMOVALS)
    assert removal_mask.sum() == 3
    removed = added.loc[~removal_mask].copy()
    _, after_remove_graph, after_remove, remove_constants = evaluate(nodes, removed)

    stages = [
        ("baseline", edges, taxonomy, baseline, constants),
        ("after_add", added, after_add_graph, after_add, add_constants),
        ("after_remove", removed, after_remove_graph, after_remove, remove_constants),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(OUT / "quadrant_coverage.csv")
    pd.Series(coverage.quadrant.value_counts(), name="count").to_csv(OUT / "quadrant_counts.csv")
    graph_data = nx.node_link_data(full)
    (OUT / "baseline_graph.json").write_text(json.dumps(graph_data, ensure_ascii=False, indent=2) + "\n")
    comparison = baseline.set_index("node_id")[["label", "is_target"]]
    structural = []
    for stage, stage_edges, graph, scores, norm in stages:
        comparison = comparison.join(scores.set_index("node_id")[STATS].add_prefix(stage + "_"))
        norm.assign(stage=stage).to_csv(OUT / f"{stage}_normalization.csv", index=False)
        stage_edges.to_csv(OUT / f"{stage}_edges.csv", index=False)
        structural.append({"stage": stage, "components": nx.number_weakly_connected_components(graph),
                           "roots": sum(d == 0 for _, d in graph.in_degree()),
                           "isolates": len(list(nx.isolates(graph))),
                           "multiple_parents": sum(d > 1 for _, d in graph.in_degree())})
    pd.DataFrame(structural).to_csv(OUT / "structure.csv", index=False)

    deltas = []
    for start, end in [("baseline", "after_add"), ("baseline", "after_remove"),
                       ("after_add", "after_remove")]:
        for metric in METRICS:
            delta = (comparison[f"{end}_{metric}"] - comparison[f"{start}_{metric}"]).round(12)
            comparison[f"delta_{end}_vs_{start}_{metric}"] = delta
            for node, change in delta[delta.abs() > 0.1].items():
                deltas.append({"node_id": node, "label": comparison.at[node, "label"],
                               "is_target": comparison.at[node, "is_target"],
                               "from": start, "to": end, "metric": metric, "delta": change})
    comparison.to_csv(OUT / "comparison.csv")
    pd.DataFrame(deltas).to_csv(OUT / "changes_gt_0_1.csv", index=False)
    print(pd.DataFrame(structural).to_string(index=False))
    print(pd.DataFrame(deltas).query("is_target == 'TRUE'").to_string(index=False))


if __name__ == "__main__":
    main()
