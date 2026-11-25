"""Slide 30 deletion scenario starting from the notebook's saved +3 graph."""
from pathlib import Path
import networkx as nx
import pandas as pd
from ha2_analysis import load_inputs, build_graphs, compute_specificity, validate_scores

ROOT = Path(__file__).resolve().parent
STAGES = ROOT / "results/perturbation"
OUT = ROOT / "results/slide30"


def run(data_dir=ROOT):
    stages = Path(data_dir) / "results/perturbation"
    out = Path(data_dir) / "results/slide30"
    nodes, baseline_edges = load_inputs(data_dir)
    added = pd.read_csv(stages / "after_add_edges.csv", keep_default_na=False)
    keys = ["source", "relation", "target"]
    added_keys = pd.MultiIndex.from_frame(added[keys])
    second_parent = ("A05", "is_a", "C14")
    assert second_parent not in pd.MultiIndex.from_frame(baseline_edges[keys])
    drop = added_keys.isin([("C04", "is_a", "C05"), ("C13", "is_a", "C01"), second_parent])
    assert drop.sum() == 3
    edges = added.loc[~drop].copy()
    _, graph = build_graphs(nodes, edges)
    scores, constants = compute_specificity(graph)
    validate_scores(graph, scores)
    out.mkdir(exist_ok=True)
    edges.to_csv(out / "after_remove_edges.csv", index=False)
    scores.to_csv(out / "after_remove_specificity.csv", index=False)
    constants.to_csv(out / "normalization.csv", index=False)
    _, pre = build_graphs(nodes, added)
    roots = {n for n, degree in pre.in_degree() if degree == 0}
    baseline = pd.read_csv(stages / "baseline_specificity.csv")
    fixed_denominator = baseline.depth_min.max()
    diagnostic = []
    for stage in ("baseline", "after_add", "after_remove", "slide30_remove"):
        if stage == "slide30_remove":
            taxonomy, values = graph, scores.copy()
        else:
            stage_edges = pd.read_csv(stages / f"{stage}_edges.csv")
            _, taxonomy = build_graphs(nodes, stage_edges)
            values = pd.read_csv(stages / f"{stage}_specificity.csv")
        distances = nx.multi_source_dijkstra_path_length(taxonomy, roots, weight=None)
        values["stage"] = stage
        values["reachable_from_pre_deletion_roots"] = values.node_id.isin(distances)
        values["coverage_fixed_roots"] = values.node_id.map(
            lambda n: "covered" if n in distances else "out_of_coverage")
        values["fixed_root_depth"] = values.node_id.map(distances)
        values["baseline_denominator"] = fixed_denominator
        values["S_depth_fixed_denominator"] = values.depth_min / fixed_denominator
        diagnostic.append(values)
    fields = ["stage", "node_id", "label", "depth_min", "depth_max",
              "max_depth_shortest", "max_depth_longest", "S_depth",
              "S_depth_range_min", "S_depth_range_max", "S_breadth",
              "reachable_from_pre_deletion_roots", "coverage_fixed_roots",
              "fixed_root_depth", "baseline_denominator", "S_depth_fixed_denominator"]
    pd.concat(diagnostic)[fields].to_csv(out / "coverage_audit.csv", index=False)
    frames = []
    for stage in ("baseline", "after_add"):
        ref = pd.read_csv(stages / f"{stage}_specificity.csv").set_index("node_id")
        cur = scores.set_index("node_id")
        for metric in ("S_depth", "S_breadth"):
            delta = (cur[metric] - ref[metric]).round(12)
            rows = cur[["label", "is_target"]].assign(reference=stage, metric=metric, delta=delta)
            frames.append(rows[delta.abs() > 0.1])
    pd.concat(frames).to_csv(out / "changes_gt_0_1.csv")


if __name__ == "__main__":
    run()
