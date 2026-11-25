"""Sequential +3/−3 experiment on the revised HA2 snapshot; never edit inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import networkx as nx
import pandas as pd

from ha2_analysis import build_graphs, compute_specificity, load_inputs, validate_scores
from ha2_coverage import baseline_identity, checked_baseline
# Retain the previous import location for callers; coverage is not rerun here.
from ha2_coverage import quadrant_coverage  # noqa: F401


ROOT = Path(__file__).resolve().parent
KEYS = ["source", "relation", "target"]
METRICS = ["S_depth", "S_breadth"]
STATS = [
    "depth_min", "depth_max", "descendants", "descendants_log1p",
    "parent_ids", "parent_count", "root_ids", "root_count", "root_path_count",
    "component_id", "component_size", "taxonomy_isolate",
    "S_depth", "S_breadth", "S_depth_longest", "S_depth_range_min", "S_depth_range_max",
]
TRANSITIONS = [
    ("baseline", "after_add"), ("baseline", "after_remove"), ("after_add", "after_remove"),
]
DELTA_THRESHOLD = 0.1
BOUNDARY_TOLERANCE = 1e-12
# Bind the experimental endpoints to identities, not recycled IDs alone.
EXPECTED_NODES = {
    "A04": ("КОЛИЧЕСТВО, ВЕЛИЧИНА, ЧИСЛО", "106515-N"),
    "A08": ("ЛИНЕЙНЫЙ РАЗМЕР", ""),
    "A05": ("ФИЗИЧЕСКАЯ СУЩНОСТЬ", "147134-N"),
    "C14": ("полость", "i110049"),
    "C23": ("эхолокация", "i36231"),
    "C04": ("расстояние", "106757-N"),
    "C13": ("место", "106451-N; i35580; i82501"),
    "C01": ("точка", "120377-N"),
    "C16": ("каверна", ""),
    "C05": ("высота", "106758-N"),
}
ADDITIONS = [
    dict(source="A04", relation="is_a", target="A08", confidence="medium",
         justification="Экспериментальная manual-гипотеза: линейный размер — количественная "
                       "величина. Присоединяет отдельную ветвь; не заявляется ребром RuWordNet."),
    dict(source="A05", relation="is_a", target="C14", confidence="high",
         justification="Экспериментальный транзитивный shortcut: в baseline уже есть "
                       "A05 → C12 → C14. Проверяет чувствительность глубины к избыточному "
                       "ребру; не заявляется прямым гиперонимом RuWordNet."),
    dict(source="C23", relation="estimates", target="C04", confidence="medium",
         justification="Экспериментальная manual-ассоциация: по задержке эха оценивается "
                       "расстояние. Нетаксономический контроль; не входит в обе оценки S."),
]
REMOVALS = [
    dict(source="C13", relation="is_a", target="C01",
         justification="Абляция внутреннего ребра место → точка: отделяет ветвь точки "
                       "и центра, не удаляя узлы."),
    dict(source="C14", relation="is_a", target="C16",
         justification="Абляция ребра к листу каверна: проверяет изоляцию листа "
                       "и потерю потомка у предков."),
    dict(source="C04", relation="is_a", target="C05",
         justification="Абляция ребра к самому глубокому листу высота: проверяет "
                       "изоляцию и смену глобального depth-знаменателя."),
]


def edge_keys(edges):
    return set(edges[KEYS].itertuples(index=False, name=None))


def large_change(delta):
    """Strict |Δ| > 0.1; suppress only floating-point equality at the boundary."""
    return abs(delta) > DELTA_THRESHOLD + BOUNDARY_TOLERANCE


def evaluate(nodes, edges):
    full, taxonomy = build_graphs(nodes, edges)
    scores, constants = compute_specificity(taxonomy)
    validate_scores(taxonomy, scores)
    # The entire non-taxonomic layer is a negative control in EVERY state.
    _, tax_only = build_graphs(nodes, edges.loc[edges.relation.eq("is_a")])
    only_scores, only_constants = compute_specificity(tax_only)
    pd.testing.assert_frame_equal(scores, only_scores)
    pd.testing.assert_frame_equal(constants, only_constants)
    return dict(full=full, taxonomy=taxonomy, scores=scores, constants=constants, edges=edges)


def prepare_states(nodes, edges):
    """Check the new scenario before any export. Return independent graph copies."""
    checks = []

    def require(name, condition):
        if not condition:
            raise ValueError(f"Perturbation precondition failed: {name}")
        checks.append(dict(check=name, passed=True))

    identities = nodes.set_index("node_id")
    for n, expected in EXPECTED_NODES.items():
        require(f"identity {n}: label + source_id",
                n in identities.index
                and tuple(identities.loc[n, ["label", "source_id"]]) == expected)
    addition_rows = [
        dict(row, provenance="manual", source_provenance="manual:perturbation")
        for row in ADDITIONS
    ]
    additions = pd.DataFrame(addition_rows).reindex(columns=edges.columns)
    removals = pd.DataFrame(REMOVALS)
    original_keys, add_keys, remove_keys = edge_keys(edges), edge_keys(additions), edge_keys(removals)
    require("exactly three distinct additions and removals", len(add_keys) == len(remove_keys) == 3)
    require("additions are absent from baseline", not original_keys & add_keys)
    require("removals are original baseline edges, not undoing additions",
            remove_keys <= original_keys and not add_keys & remove_keys)
    require("all endpoints exist", {
        n for key in add_keys | remove_keys for n in (key[0], key[2])
    } <= set(nodes.node_id))
    baseline = evaluate(nodes, edges.copy(deep=True))
    tax, s = baseline["taxonomy"], baseline["scores"].set_index("node_id")
    require("A08 is an observed root before its bridge", tax.in_degree("A08") == 0)
    require("bridge joins different taxonomy components",
            not nx.has_path(tax.to_undirected(), "A04", "A08"))
    require("A05 → C14 is already implied transitively",
            nx.has_path(tax, "A05", "C14") and not tax.has_edge("A05", "C14"))
    require("C13 → C01 is internal and C01 has one parent",
            tax.in_degree("C01") == 1 and tax.out_degree("C01") > 0)
    require("C16 is a leaf with one parent",
            tax.in_degree("C16") == 1 and tax.out_degree("C16") == 0)
    require("C05 is a deepest leaf with one parent",
            tax.in_degree("C05") == 1 and tax.out_degree("C05") == 0
            and s.loc["C05", "depth_min"] == s.depth_min.max())
    added = pd.concat([edges, additions], ignore_index=True)
    after_add = evaluate(nodes, added)
    removal_mask = pd.MultiIndex.from_frame(added[KEYS]).isin(list(remove_keys))
    require("exactly three edges removed from after_add", int(removal_mask.sum()) == 3)
    after_remove = evaluate(nodes, added.loc[~removal_mask].reset_index(drop=True))
    stages = dict(baseline=baseline, after_add=after_add, after_remove=after_remove)
    require("same node inventory in all states",
            all(set(state["full"]) == set(nodes.node_id) for state in stages.values()))
    require("typed edge counts baseline, +3, −3",
            [len(stages[k]["edges"]) for k in stages] == [len(edges), len(edges) + 3, len(edges)])
    require("added cavity parent creates a nondegenerate depth interval",
            after_add["scores"].set_index("node_id").loc["C14", "depth_min"]
            < after_add["scores"].set_index("node_id").loc["C14", "depth_max"])
    checks.append(dict(check="DAG, all-node scores, non-taxonomic control in all states", passed=True))
    return stages, checks, addition_rows


def compare_states(stages):
    """All transitions, plus a raw-structure/denominator decomposition."""
    baseline = stages["baseline"]["scores"].set_index("node_id")
    comparison = baseline[["label", "is_target"]].copy()
    for name, state in stages.items():
        comparison = comparison.join(state["scores"].set_index("node_id")[STATS].add_prefix(name + "_"))
    rows = []
    for start, end in TRANSITIONS:
        a = stages[start]["scores"].set_index("node_id")
        b = stages[end]["scores"].set_index("node_id")
        ca = stages[start]["constants"].set_index("scope").loc["global"]
        cb = stages[end]["constants"].set_index("scope").loc["global"]
        for metric in METRICS:
            difference = b[metric] - a[metric]  # Do not round before selection.
            comparison[f"delta_{end}_vs_{start}_{metric}"] = difference
            raw = "depth_min" if metric == "S_depth" else "descendants"
            denominator = "max_depth_shortest" if metric == "S_depth" else "breadth_log_denominator"
            da, db = float(ca[denominator]), float(cb[denominator])
            sign = 1 if metric == "S_depth" else -1
            for n, delta in difference.items():
                va, vb = float(a.loc[n, raw]), float(b.loc[n, raw])
                fa, fb = (va, vb) if metric == "S_depth" else (math.log1p(va), math.log1p(vb))
                structure = sign * (fb - fa) / da
                normalization = sign * fb * (1 / db - 1 / da)
                if not math.isclose(structure + normalization, delta, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Delta decomposition does not sum to the observed change")
                rows.append(dict(
                    node_id=n, label=a.loc[n, "label"], is_target=a.loc[n, "is_target"],
                    **{"from": start, "to": end}, metric=metric,
                    S_from=float(a.loc[n, metric]), S_to=float(b.loc[n, metric]),
                    delta=float(delta), abs_delta=abs(float(delta)),
                    raw_from=int(va), raw_to=int(vb),
                    denominator_from=da, denominator_to=db,
                    delta_structure_from_scale=structure, delta_normalization=normalization,
                    exceeds_threshold=large_change(delta),
                ))
    deltas = pd.DataFrame(rows)
    return comparison, deltas


def run_perturbation(data_dir=ROOT):
    data_dir = Path(data_dir)
    reference, manifest = checked_baseline(data_dir)
    nodes, edges = load_inputs(data_dir)
    stages, checks, additions = prepare_states(nodes, edges)
    # Recompute the entire baseline table, not merely the two S columns.
    for frame, saved in (
        (stages["baseline"]["scores"], reference),
        (stages["baseline"]["constants"], pd.read_csv(data_dir / "results/normalization.csv")),
    ):
        pd.testing.assert_frame_equal(frame, saved, check_dtype=False, check_exact=False,
                                      rtol=1e-12, atol=1e-12)
    comparison, deltas = compare_states(stages)
    large = deltas.loc[deltas.exceeds_threshold].copy()
    structures = []
    for name, state in stages.items():
        t = state["taxonomy"]
        structures.append(dict(
            stage=name, node_count=len(t), typed_edges=len(state["edges"]),
            taxonomy_edges=t.number_of_edges(), components=nx.number_weakly_connected_components(t),
            roots=sum(d == 0 for _, d in t.in_degree()), isolates=len(list(nx.isolates(t))),
            isolate_ids=";".join(sorted(nx.isolates(t))),
            multiple_parents=sum(d > 1 for _, d in t.in_degree()),
            unequal_depth_nodes=int(state["scores"].depth_min.ne(state["scores"].depth_max).sum()),
        ))
    change_summary = []
    for start, end in TRANSITIONS:
        for metric in METRICS:
            selected = large.loc[large["from"].eq(start) & large.to.eq(end) & large.metric.eq(metric)]
            change_summary.append(dict(
                **{"from": start, "to": end}, metric=metric,
                all_nodes_exceeding=int(len(selected)),
                target_nodes_exceeding=int(selected.is_target.eq("TRUE").sum()),
            ))
    scenario = dict(
        design="sequential: baseline → add three → remove three from after_add",
        experimental_only=True, not_ontology_corrections=True,
        identities={n: dict(label=v[0], source_id=v[1]) for n, v in EXPECTED_NODES.items()},
        additions=additions, removals=REMOVALS,
        delta_threshold=DELTA_THRESHOLD, floating_boundary_tolerance=BOUNDARY_TOLERANCE,
        primary_normalization="global over the same 44 nodes, recalculated in every state",
        non_taxonomic_edges_used_for_S=0,
    )
    # Freeze byte-exact baseline files. The original results/ directory is not replaced.
    snapshots = {
        "baseline_nodes.csv": data_dir / "nodes.csv",
        "baseline_edges.csv": data_dir / "edges.csv",
        "baseline_graph.json": data_dir / "results/typed_graph.json",
        "baseline_taxonomy_graph.json": data_dir / "results/taxonomy_graph.json",
        "baseline_specificity.csv": data_dir / "results/specificity.csv",
        "baseline_normalization.csv": data_dir / "results/normalization.csv",
    }
    payloads = {name: path.read_bytes() for name, path in snapshots.items()}

    def csv_payload(name, table, *, index=False):
        payloads[name] = table.to_csv(
            index=index, float_format="%.15g", lineterminator="\n"
        ).encode("utf-8")

    def json_payload(name, value):
        payloads[name] = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    for name in ("after_add", "after_remove"):
        state = stages[name]
        for suffix, table in (
            ("edges", state["edges"]), ("specificity", state["scores"]),
            ("normalization", state["constants"]),
        ):
            csv_payload(f"{name}_{suffix}.csv", table)
        json_payload(f"{name}_graph.json", nx.node_link_data(state["full"]))
        json_payload(f"{name}_taxonomy_graph.json", nx.node_link_data(state["taxonomy"]))
    csv_payload("comparison.csv", comparison, index=True)
    csv_payload("deltas.csv", deltas)
    csv_payload("changes_gt_0_1.csv", large)
    csv_payload("change_summary.csv", pd.DataFrame(change_summary))
    csv_payload("structure.csv", pd.DataFrame(structures))
    json_payload("scenario.json", scenario)
    json_payload("preconditions.json", dict(all_passed=True, checks=checks))
    experiment_manifest = dict(
        run_kind="perturbation", source_commit=manifest["source_commit"],
        baseline_identity_sha256=baseline_identity(manifest),
        baseline_output_sha256=manifest["baseline_output_sha256"],
        input_sha256=manifest["input_sha256"],
        code_sha256={
            **manifest["code_sha256"],
            "ha2_coverage.py": hashlib.sha256((ROOT / "ha2_coverage.py").read_bytes()).hexdigest(),
            "ha2_perturbation.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        python=sys.version.split()[0],
        packages={p: manifest["packages"][p] for p in ("pandas", "networkx")},
        scenario="scenario.json",
        saved_baseline_sha256={
            name: hashlib.sha256(payloads[name]).hexdigest() for name in snapshots
        },
        baseline_preserved=True, source_inputs_modified=False,
        non_taxonomic_control_passed_all_states=True,
        interpretation_written=False,
        output_sha256={name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
    )
    json_payload("run_manifest.json", experiment_manifest)
    output = data_dir / "results/perturbation"
    output.mkdir(exist_ok=True)
    for name, value in payloads.items():
        (output / name).write_bytes(value)
    manifest["perturbation_status"] = "computed"
    manifest["perturbation_performed"] = True
    manifest["perturbation_manifest"] = "perturbation/run_manifest.json"
    (data_dir / "results/run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return dict(
        output="results/perturbation", design=scenario["design"],
        structure=structures, changes_gt_0_1=change_summary,
        baseline_preserved=True, interpretation_written=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run_perturbation(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
