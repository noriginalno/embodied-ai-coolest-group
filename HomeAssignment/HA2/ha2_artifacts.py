"""Rebuild and fingerprint the complete current HA2 artifact set, excluding archives."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import pandas as pd

import ha2_analysis as analysis
from role4_slide30 import run as slide30
from ha2_coverage import QUADRANTS, baseline_identity, checked_baseline, run_coverage
from ha2_perturbation import run_perturbation


ROOT = Path(__file__).resolve().parent
FIGURE_STEMS = [
    "taxonomy", "non_taxonomic", "ha1_quadrants", "perturbation_deltas",
    "perturbation/after_add_taxonomy", "perturbation/after_add_non_taxonomic",
    "perturbation/after_remove_taxonomy", "perturbation/after_remove_non_taxonomic",
]
CODE_FILES = ["sync.py", "ha2_analysis.py", "ha2_coverage.py", "ha2_perturbation.py", "ha2_artifacts.py", "role4_slide30.py"]
DOC_FILES = ["README.md", "ROLE4.md", "INPUT_FORMAT.md", "INTERPRETATION.md"]
SLIDE30 = ["results/slide30/" + name for name in ("after_remove_edges.csv", "after_remove_specificity.csv", "normalization.csv", "changes_gt_0_1.csv", "coverage_audit.csv")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def notebook_source_sha256(path):
    cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
    return hashlib.sha256(json.dumps([
        dict(cell_type=c["cell_type"], source="".join(c["source"])) for c in cells
    ], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def checked_computations(data_dir):
    data_dir = Path(data_dir)
    _, baseline = checked_baseline(data_dir)
    if baseline["coverage_status"] != "computed" or baseline["perturbation_status"] != "computed":
        raise ValueError("Recompute coverage and perturbation before building artifacts")
    paths = [f"results/{name}" for name in baseline["baseline_output_sha256"]]
    paths.append("results/run_manifest.json")
    for name in ("ha1_coverage", "perturbation"):
        folder = data_dir / "results" / name
        manifest = json.loads((folder / "run_manifest.json").read_text())
        if manifest["baseline_identity_sha256"] != baseline_identity(baseline):
            raise ValueError(f"Stale {name} baseline identity")
        for path, digest in manifest["input_sha256"].items():
            if sha256(data_dir / path) != digest:
                raise ValueError(f"Stale {name} input: {path}")
        code_hashes = manifest.get("code_sha256", {
            "ha2_coverage.py": manifest.get("coverage_code_sha256"),
        })
        for path, digest in code_hashes.items():
            if sha256(ROOT / path) != digest:
                raise ValueError(f"Stale {name} code: {path}")
        for path, digest in manifest["output_sha256"].items():
            if sha256(folder / path) != digest:
                raise ValueError(f"Stale {name} output: {path}")
            paths.append(f"results/{name}/{path}")
        paths.append(f"results/{name}/run_manifest.json")
    return baseline, paths


def draw_quadrants(data_dir):
    folder = Path(data_dir) / "results/ha1_coverage"
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for offset, suffix, label, color in (
        (-.19, "", "Исходная разметка", "#167c80"),
        (.19, "_social0", "social=0 у «близости»", "#b06b00"),
    ):
        counts = pd.read_csv(folder / f"quadrant_counts{suffix}.csv").set_index("quadrant")
        bars = ax.bar([i + offset for i in range(4)], counts.loc[QUADRANTS, "count"],
                      width=.36, label=label, color=color)
        ax.bar_label(bars, padding=3)
    ax.axhline(3, color="#9d3445", linestyle="--", label="Минимум HA1: 3")
    ax.set(xticks=range(4), xticklabels=QUADRANTS, ylim=(0, 10),
           ylabel="Число целевых понятий",
           title="HA1 · C×S по 25 целевым понятиям\n"
                 "Отдельная min–max-нормировка; граница 0.5 включается в «+»")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", ncols=2)
    return fig


def draw_deltas(data_dir):
    data = pd.read_csv(Path(data_dir) / "results/perturbation/comparison.csv",
                       dtype={"is_target": str})
    data = data.loc[data.is_target.eq("TRUE")].sort_values("node_id")
    fig, axes = plt.subplots(1, 2, figsize=(15, 12), sharey=True, layout="constrained")
    for ax, metric in zip(axes, ("S_depth", "S_breadth")):
        for offset, stage, label, color in (
            (-.18, "after_add", "После +3", "#167c80"),
            (.18, "after_remove", "После −3 (последовательно)", "#a54519"),
        ):
            ax.barh([i + offset for i in range(len(data))],
                    data[f"delta_{stage}_vs_baseline_{metric}"], height=.32,
                    label=label, color=color)
        ax.axvline(0, color="#42586f", linewidth=.8)
        for boundary in (-.1, .1):
            ax.axvline(boundary, color="#9d3445", linestyle="--", linewidth=.8)
        ax.set(title=metric, xlabel="Δ относительно baseline")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(range(len(data)), data.node_id + " · " + data.label)
    axes[0].invert_yaxis()
    fig.suptitle("HA2 · изменения обеих S у 25 целевых понятий\n"
                 "Знаменатели пересчитаны для каждого состояния; пунктир = ±0.1", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    return fig


def render_artifacts(data_dir=ROOT):
    """Render every current graph from its exported JSON, not stale in-memory state."""
    data_dir = Path(data_dir)
    checked_computations(data_dir)
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        for stage, label in (("baseline", "Baseline"), ("after_add", "После +3"), ("after_remove", "После −3")):
            if stage == "baseline":
                graph_paths = ["results/taxonomy_graph.json", "results/typed_graph.json"]
                stems = ["taxonomy", "non_taxonomic"]
            else:
                graph_paths = [f"results/perturbation/{stage}_{suffix}.json"
                               for suffix in ("taxonomy_graph", "graph")]
                stems = [f"perturbation/{stage}_{suffix}" for suffix in ("taxonomy", "non_taxonomic")]
            graphs = [nx.node_link_graph(json.loads((data_dir / p).read_text()), edges="edges") for p in graph_paths]
            for index, (graph, stem, draw) in enumerate(zip(
                graphs, stems, (analysis.draw_taxonomy, analysis.draw_non_taxonomic)
            )):
                taxonomy_title = (
                    f"HA2 · {label} · таксономия: {len(graph)} узла, {graph.number_of_edges()} рёбер\n"
                    "is_a: родитель → потомок; вертикальный ранг = максимальная глубина."
                )
                figure = draw(graph, title=taxonomy_title) if stage != "baseline" and index == 0 else draw(graph)
                try:
                    if stage != "baseline" and index == 1:
                        figure.suptitle(
                            f"HA2 · {label} · нетаксономические связи\n"
                            "Все показанные отношения исключены из расчётов S.", fontsize=14,
                        )
                    analysis.save_figure(figure, data_dir / "figures" / stem)
                finally:
                    plt.close(figure)
        for stem, draw in (("ha1_quadrants", draw_quadrants), ("perturbation_deltas", draw_deltas)):
            figure = draw(data_dir)
            try:
                analysis.save_figure(figure, data_dir / "figures" / stem)
            finally:
                plt.close(figure)
    return [f"figures/{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "svg")]


def write_artifact_manifest(data_dir=ROOT):
    data_dir = Path(data_dir)
    baseline, paths = checked_computations(data_dir)
    source_hash = notebook_source_sha256(data_dir / "HA2_Semantic_Network.ipynb")
    if source_hash != baseline["notebook_source_sha256"]:
        raise ValueError("Notebook sources changed after the baseline run")
    baseline["interpretation_written"] = True
    baseline["artifacts_status"] = "computed"
    baseline["artifacts_manifest"] = "artifact_manifest.json"
    (data_dir / "results/run_manifest.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths += [f"figures/{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "svg")]
    slide30(data_dir)
    paths += DOC_FILES + SLIDE30
    manifest = dict(
        run_kind="complete_artifacts", source_commit=baseline["source_commit"],
        baseline_identity_sha256=baseline_identity(baseline),
        input_sha256=baseline["input_sha256"],
        code_sha256={name: sha256(ROOT / name) for name in CODE_FILES},
        notebook_source_sha256=source_hash,
        notebook_hash_policy="cell types and sources only; outputs and execution timestamps excluded",
        current_output_sha256={name: sha256(data_dir / name) for name in sorted(paths)},
        archive_policy="archive/ is excluded; PR5 alternatives are preserved there verbatim",
        artifacts_status="computed", interpretation_written=True,
    )
    (data_dir / "results/artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_artifacts(data_dir=ROOT):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "results/artifact_manifest.json").read_text())
    baseline, paths = checked_computations(data_dir)
    if baseline.get("artifacts_status") != "computed":
        raise ValueError("Artifact integration is pending")
    expected = set(paths + DOC_FILES + SLIDE30 + [
        f"figures/{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "svg")
    ])
    if set(manifest["current_output_sha256"]) != expected:
        raise ValueError("Artifact inventory is incomplete")
    for group, folder in (("input_sha256", data_dir), ("current_output_sha256", data_dir), ("code_sha256", ROOT)):
        for name, digest in manifest[group].items():
            if sha256(folder / name) != digest:
                raise ValueError(f"Artifact fingerprint mismatch: {name}")
    if manifest["baseline_identity_sha256"] != baseline_identity(baseline):
        raise ValueError("Artifact baseline identity mismatch")
    if manifest["notebook_source_sha256"] != notebook_source_sha256(data_dir / "HA2_Semantic_Network.ipynb"):
        raise ValueError("Artifact notebook sources changed")
    return dict(verified_files=len(expected), figure_files=len(FIGURE_STEMS) * 2,
                source_commit=manifest["source_commit"], interpretation_written=True)


def run_artifacts(data_dir=ROOT):
    data_dir = Path(data_dir)
    nodes, edges = analysis.load_inputs(data_dir)
    full, taxonomy = analysis.build_graphs(nodes, edges)
    scores, constants = analysis.compute_specificity(taxonomy)
    analysis.validate_scores(taxonomy, scores)
    audit = analysis.audit_directions(data_dir, nodes, edges)
    if not audit.source_direction_confirmed.all():
        raise ValueError("Unconfirmed input edge directions")
    analysis.export_results(data_dir, nodes, edges, full, taxonomy, scores, constants, audit)
    run_coverage(data_dir)
    run_perturbation(data_dir)
    render_artifacts(data_dir)
    write_artifact_manifest(data_dir)
    return verify_artifacts(data_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="Verify without recalculation or writes")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    result = verify_artifacts(args.data_dir) if args.check else run_artifacts(args.data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
