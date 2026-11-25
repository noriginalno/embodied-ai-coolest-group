"""Slide 30 deletion scenario starting from the notebook's saved +3 graph."""
from pathlib import Path
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
