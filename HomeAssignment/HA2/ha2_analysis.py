"""HA2 role 3: typed graphs, explicitly scoped specificity, and figures.

No network access; nodes.csv and edges.csv are inputs and are never rewritten.
The same functions are used by the notebook and can be reused by role 4.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
import networkx as nx
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
MIN_NORMALIZATION_NODES = 10
RELATION_COLORS = {
    "is_a": "#42586f",
    "specifies_location_of": "#167c80",
    "has_endpoint": "#a54519",
    "passes_through": "#7851a9",
}
TARGET_COLOR = "#dcedfa"
AUXILIARY_COLOR = "#edf0f2"
ROOT_COLOR = "#ad6b17"


def load_inputs(data_dir=DATA_DIR):
    """Read the role 2 snapshot and add transparent, non-semantic metadata."""
    data_dir = Path(data_dir)
    nodes = pd.read_csv(data_dir / "nodes.csv", dtype=str, keep_default_na=False)
    edges = pd.read_csv(data_dir / "edges.csv", dtype=str, keep_default_na=False)
    annotations = pd.read_csv(
        data_dir / "pos_annotations.csv", dtype=str, keep_default_na=False
    )
    if not annotations.node_id.is_unique:
        raise ValueError("Duplicate POS annotations")
    extra = set(annotations.node_id) - set(nodes.node_id)
    if extra:
        raise ValueError(f"POS annotations for absent nodes: {sorted(extra)}")
    by_id = annotations.set_index("node_id")
    pos, provenance = [], []
    for row in nodes.itertuples(index=False):
        suffixes = set(re.findall(r"\d+-(N|V|A)(?=;|$)", row.source_id))
        if row.node_id in by_id.index:
            pos.append(by_id.loc[row.node_id, "POS"])
            provenance.append(by_id.loc[row.node_id, "provenance"] + ":pos_annotations.csv")
        elif len(suffixes) == 1:
            pos.append({"N": "NOUN", "V": "VERB", "A": "ADJ"}[suffixes.pop()])
            provenance.append("wordnet:source_id_suffix")
        else:
            pos.append("UNKNOWN")
            provenance.append("not_supplied")
    nodes["POS"] = pos
    nodes["POS_provenance"] = provenance
    # Definitions are copied, not generated; auxiliary definitions remain absent.
    concepts = pd.read_csv(
        data_dir / "raw/Concepts_hw2.csv", dtype=str, keep_default_na=False
    )
    if not concepts.concept_id.is_unique:
        raise ValueError("Duplicate concept IDs in raw/Concepts_hw2.csv")
    definitions = concepts.set_index("concept_id")
    nodes["definition"] = nodes.node_id.map(definitions.definition).fillna("")
    nodes["definition_provenance"] = nodes.node_id.map(
        definitions.provenance
    ).fillna("not_supplied")
    return nodes, edges


def build_graphs(nodes, edges):
    """Return a lossless MultiDiGraph and an all-node, parent→child DAG."""
    required_nodes = {"node_id", "label", "node_type", "is_target"}
    required_edges = {
        "source", "relation", "target", "provenance", "confidence", "justification"
    }
    if not required_nodes <= set(nodes.columns) or not required_edges <= set(edges.columns):
        raise ValueError("Missing required node or edge columns")
    if nodes.empty or not nodes.node_id.is_unique:
        raise ValueError("Node IDs must be nonempty and unique")
    if nodes[list(required_nodes)].eq("").any().any():
        raise ValueError("Empty required node field")
    if not set(nodes.is_target) <= {"TRUE", "FALSE"}:
        raise ValueError("is_target must be TRUE or FALSE")
    if edges[list(required_edges)].eq("").any().any():
        raise ValueError("Empty required edge field")
    if edges.duplicated(["source", "relation", "target"]).any():
        raise ValueError("Duplicate typed edges")
    missing = (set(edges.source) | set(edges.target)) - set(nodes.node_id)
    if missing:
        raise ValueError(f"Unknown edge endpoints: {sorted(missing)}")
    full = nx.MultiDiGraph(
        taxonomy_direction="parent_to_child",
        note="An is_a edge u→v means: v is a kind of u.",
    )
    for row in nodes.to_dict("records"):
        full.add_node(row["node_id"], **{k: v for k, v in row.items() if k != "node_id"})
    for row in edges.to_dict("records"):
        full.add_edge(
            row["source"], row["target"], key=row["relation"],
            **{k: v for k, v in row.items() if k not in {"source", "target"}},
        )
    taxonomy = nx.DiGraph(taxonomy_direction="parent_to_child")
    # Add nodes FIRST: constructing a graph only from edges loses isolates.
    taxonomy.add_nodes_from(full.nodes(data=True))
    taxonomy.add_edges_from(
        (u, v, dict(d)) for u, v, d in full.edges(data=True)
        if d["relation"] == "is_a"
    )
    if not nx.is_directed_acyclic_graph(taxonomy):
        raise ValueError(f"is_a cycle: {nx.find_cycle(taxonomy)}")
    if set(full) != set(taxonomy) or full.number_of_edges() != len(edges):
        raise ValueError("Graph construction lost nodes or typed edges")
    return full, taxonomy


def audit_directions(data_dir, nodes, edges):
    """Check storage orientation against source parent/child columns, not meaning."""
    data_dir = Path(data_dir)
    raw = pd.read_csv(data_dir / "raw/Edges.csv", dtype=str, keep_default_na=False)
    aliases = {"A24": "C01", "A01": "C18", "A28": "C18"}
    evidence = {
        (aliases.get(r.parent, r.parent), aliases.get(r.child, r.child)):
        "raw/Edges.csv: parent -> child (role 2 aliases)"
        for r in raw.itertuples(index=False) if r.relation == "is_a"
    }
    ha1_path = data_dir.parent / "HA1_submission/data/ruwordnet_edges.csv"
    synset_nodes = {}
    for row in nodes.itertuples(index=False):
        for synset in row.source_id.split("; "):
            if re.fullmatch(r"\d+-[NAV]", synset):
                synset_nodes[synset] = row.node_id
    if ha1_path.exists():
        ha1 = pd.read_csv(ha1_path, dtype=str, keep_default_na=False)
        for r in ha1.itertuples(index=False):
            if r.parent_id in synset_nodes and r.child_id in synset_nodes:
                evidence[synset_nodes[r.parent_id], synset_nodes[r.child_id]] = (
                    "HA1_submission/data/ruwordnet_edges.csv: parent_id -> child_id"
                )
    rows = []
    for r in edges.loc[edges.relation.eq("is_a")].itertuples(index=False):
        support = evidence.get((r.source, r.target), "")
        rows.append({
            "source": r.source, "relation": r.relation, "target": r.target,
            "direction": "parent_to_child", "source_direction_confirmed": bool(support),
            "evidence": support or "not checked against a source parent/child table",
        })
    return pd.DataFrame(rows)


def taxonomy_structure(taxonomy):
    """Shortest/longest paths from ANY observed root; no invented super-root."""
    if not nx.is_directed_acyclic_graph(taxonomy):
        raise ValueError("Specificity requires an is_a DAG")
    result = {}
    components = sorted(
        nx.weakly_connected_components(taxonomy),
        key=lambda c: (-len(c), tuple(sorted(c))),
    )
    membership = {
        n: (f"K{i:02d}", len(c))
        for i, c in enumerate(components, 1) for n in c
    }
    for n in nx.lexicographical_topological_sort(taxonomy):
        parents = sorted(taxonomy.predecessors(n))
        if parents:
            d_min = 1 + min(result[p]["depth_min"] for p in parents)
            d_max = 1 + max(result[p]["depth_max"] for p in parents)
            roots = set().union(*(result[p]["roots"] for p in parents))
            path_count = sum(result[p]["root_path_count"] for p in parents)
        else:
            d_min, d_max, roots, path_count = 0, 0, {n}, 1
        component_id, component_size = membership[n]
        result[n] = {
            "depth_min": d_min, "depth_max": d_max, "roots": roots,
            "root_path_count": path_count, "parent_ids": ";".join(parents),
            "parent_count": len(parents), "component_id": component_id,
            "component_size": component_size,
            "descendants": len(nx.descendants(taxonomy, n)),
        }
    return result


def compute_specificity(taxonomy, min_normalization_nodes=MIN_NORMALIZATION_NODES):
    """Return all-node scores and every normalization constant.

    Main estimates exactly follow the instructor notebook (global scope).
    The depth interval uses a separate common longest-depth denominator.
    POS-restricted estimates and slide 28's contradictory formula are diagnostic.
    Undefined or undersized normalizations produce NaN, never fabricated zeros.
    """
    structure = taxonomy_structure(taxonomy)
    if not structure:
        raise ValueError("Cannot score an empty taxonomy")
    groups = {"global": taxonomy}
    pos_values = {taxonomy.nodes[n].get("POS", "UNKNOWN") for n in taxonomy}
    for pos in sorted(pos_values - {"UNKNOWN"}):
        groups[f"POS:{pos}"] = taxonomy.subgraph(
            [n for n in taxonomy if taxonomy.nodes[n].get("POS") == pos]
        ).copy()
    scoped_structure = {"global": structure}
    constants = {}
    for scope, graph in groups.items():
        stats = structure if scope == "global" else taxonomy_structure(graph)
        scoped_structure[scope] = stats
        size = len(graph)
        max_short = max(s["depth_min"] for s in stats.values())
        max_long = max(s["depth_max"] for s in stats.values())
        max_desc = max(s["descendants"] for s in stats.values())
        alternate = [
            s["depth_min"] + math.log((1 + s["descendants"]) / size)
            for s in stats.values()
        ]
        constants[scope] = {
            "scope": scope, "node_count": size, "edge_count": graph.number_of_edges(),
            "max_depth_shortest": max_short, "max_depth_longest": max_long,
            "max_descendants": max_desc,
            "breadth_log_denominator": math.log1p(max_desc),
            "slide28_raw_min": min(alternate), "slide28_raw_max": max(alternate),
            "min_normalization_nodes": min_normalization_nodes,
            "normalization_status": (
                "raw_only_small_graph" if size < min_normalization_nodes
                else "zero_depth_or_breadth_possible" if not max_short or not max_desc
                else "ok"
            ),
        }

    def ratio(value, denominator, c):
        if c["node_count"] < min_normalization_nodes or denominator == 0:
            return math.nan
        return value / denominator

    rows = []
    global_c = constants["global"]
    for n in sorted(taxonomy):
        s = structure[n]
        attrs = taxonomy.nodes[n]
        pos = attrs.get("POS", "UNKNOWN")
        scope = f"POS:{pos}"
        row = {
            "node_id": n, "label": attrs.get("label", n),
            "node_type": attrs.get("node_type", ""),
            "is_target": attrs.get("is_target", ""),
            "POS": pos, "POS_provenance": attrs.get("POS_provenance", "not_supplied"),
            **{k: v for k, v in s.items() if k != "roots"},
            "root_ids": ";".join(sorted(s["roots"])),
            "root_count": len(s["roots"]),
            "is_root": taxonomy.in_degree(n) == 0,
            "taxonomy_isolate": taxonomy.degree(n) == 0,
            "coverage_status": (
                "isolated_in_snapshot" if taxonomy.degree(n) == 0
                else "observed_taxonomy_only"
            ),
            "depth_range": f'{s["depth_min"]}..{s["depth_max"]}',
            "descendants_log1p": math.log1p(s["descendants"]),
        }
        for key in ("max_depth_shortest", "max_depth_longest", "max_descendants",
                    "breadth_log_denominator", "normalization_status"):
            row[key] = global_c[key]
        row["S_depth"] = ratio(s["depth_min"], global_c["max_depth_shortest"], global_c)
        row["S_depth_longest"] = ratio(s["depth_max"], global_c["max_depth_longest"], global_c)
        row["S_depth_range_min"] = ratio(s["depth_min"], global_c["max_depth_longest"], global_c)
        row["S_depth_range_max"] = row["S_depth_longest"]
        breadth_ratio = ratio(
            row["descendants_log1p"], global_c["breadth_log_denominator"], global_c
        )
        row["S_breadth"] = 1 - breadth_ratio
        if scope in constants:
            p = scoped_structure[scope][n]
            c = constants[scope]
            row.update({
                "pos_node_count": c["node_count"], "pos_depth_min": p["depth_min"],
                "pos_depth_max": p["depth_max"], "pos_descendants": p["descendants"],
                "normalization_status_pos": c["normalization_status"],
                "S_depth_pos": ratio(p["depth_min"], c["max_depth_shortest"], c),
                "S_breadth_pos": 1 - ratio(
                    math.log1p(p["descendants"]), c["breadth_log_denominator"], c
                ),
                "slide28_raw_min": p["depth_min"] + math.log((1 + p["descendants"]) / c["node_count"]),
                "slide28_raw_max": p["depth_max"] + math.log((1 + p["descendants"]) / c["node_count"]),
            })
            row["slide28_minmax_pos"] = ratio(
                row["slide28_raw_min"] - c["slide28_raw_min"],
                c["slide28_raw_max"] - c["slide28_raw_min"], c,
            )
        else:
            row["normalization_status_pos"] = "unknown_POS"
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(constants.values())


def validate_scores(taxonomy, scores):
    """Check arithmetic invariants without asserting that the ontology is true."""
    if set(scores.node_id) != set(taxonomy) or not scores.node_id.is_unique:
        raise ValueError("Score table must cover every node exactly once")
    if (scores.depth_min > scores.depth_max).any():
        raise ValueError("Invalid depth interval")
    score_columns = [c for c in scores if c.startswith("S_")] + ["slide28_minmax_pos"]
    for col in score_columns:
        if col not in scores:
            continue
        values = scores[col].dropna()
        if not values.between(-1e-12, 1 + 1e-12).all():
            raise ValueError(f"Out-of-range normalized values in {col}")
    indexed = scores.set_index("node_id")
    # Longest ranks, unlike shortest paths, must increase on EVERY category edge.
    for parent, child in taxonomy.edges:
        if indexed.loc[child, "depth_max"] <= indexed.loc[parent, "depth_max"]:
            raise ValueError("Parent→child hierarchy rank is not increasing")


def export_results(data_dir, nodes, edges, full, taxonomy, scores, constants, audit):
    """Write results/snapshots only; never modify the authoritative input CSVs."""
    data_dir = Path(data_dir)
    output = data_dir / "results"
    output.mkdir(exist_ok=True)
    targets = scores.loc[scores.is_target.eq("TRUE")].copy()
    for name, frame in (
        ("specificity.csv", scores), ("specificity_targets.csv", targets),
        ("normalization.csv", constants), ("direction_audit.csv", audit),
        ("node_attributes.csv", nodes),
    ):
        frame.to_csv(output / name, index=False, float_format="%.15g", lineterminator="\n")
    for name, graph in (("typed_graph.json", full), ("taxonomy_graph.json", taxonomy)):
        (output / name).write_text(
            json.dumps(nx.node_link_data(graph), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    checks = {
        "source_commit": "5ca95838c2b38ae84cf26a23cdb1cf9be71ac4b6",
        "node_count": len(nodes), "target_count": len(targets),
        "auxiliary_count": int(nodes.node_type.eq("auxiliary").sum()),
        "typed_edge_count": len(edges), "relation_counts": edges.relation.value_counts().to_dict(),
        "taxonomy_node_count": len(taxonomy), "taxonomy_edge_count": taxonomy.number_of_edges(),
        "taxonomy_is_DAG": nx.is_directed_acyclic_graph(taxonomy),
        "taxonomy_weak_components": nx.number_weakly_connected_components(taxonomy),
        "taxonomy_root_count": sum(taxonomy.in_degree(n) == 0 for n in taxonomy),
        "taxonomy_isolates": sorted(nx.isolates(taxonomy)),
        "multiple_parent_nodes": sorted(n for n in taxonomy if taxonomy.in_degree(n) > 1),
        "all_nodes_scored": set(scores.node_id) == set(nodes.node_id),
        "source_directions_confirmed": int(audit.source_direction_confirmed.sum()),
        "taxonomy_direction": "parent_to_child; u -> v means v is_a u",
        "non_taxonomic_edges_used_for_specificity": 0,
        "warnings": [
            "Input has 56 nodes, not 15–25. Inventory is retained without review by role 3.",
            "Observed roots/leaves/isolates are properties of this snapshot, not all of RuWordNet.",
            "POS is supplemented manually where a synset suffix is unavailable; see pos_annotations.csv.",
            "ADP and ADV subgraphs have fewer than 10 nodes: normalized POS values are undefined.",
            "Primary global scores for isolates are mechanical values, not evidence of lexical specificity.",
            "Alternative slide 28 formula and POS normalization are diagnostic, not primary scores.",
        ],
    }
    (output / "validation.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths = [
        data_dir / "nodes.csv", data_dir / "edges.csv", data_dir / "pos_annotations.csv",
        data_dir / "raw/Concepts_hw2.csv", data_dir / "raw/Edges.csv",
        data_dir / "ha2_analysis.py",
        data_dir.parent / "HA1_submission/data/ruwordnet_edges.csv",
        data_dir.parents[1] / "Seminars/Class03/Class_03_Instructor_Working_Notebook.ipynb",
        data_dir.parents[1] / "Seminars/Class03/Class_03_Semantic_Networks_Lecture_and_Seminar_Expanded_Lecture.pptx",
    ]
    manifest = {
        "source_commit": checks["source_commit"], "python": sys.version.split()[0],
        "packages": {p: importlib.metadata.version(p) for p in (
            "numpy", "pandas", "matplotlib", "networkx", "nbformat", "nbclient", "ipykernel"
        )},
        "input_sha256": {
            str(p.relative_to(data_dir.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.exists()
        },
        "primary_normalization": "global over all supplied nodes, not separately by component",
        "normalization_constants": "normalization.csv",
        "interpretation_written": False, "perturbation_performed": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return checks


def layered_positions(taxonomy):
    """Deterministic longest-depth ranks; no Graphviz/system binary required."""
    stats = taxonomy_structure(taxonomy)
    layers = {}
    for n in taxonomy:
        layers.setdefault(stats[n]["depth_max"], []).append(n)
    positions = {}
    for depth, layer in sorted(layers.items()):
        def order_key(n):
            parent_xs = [positions[p][0] for p in taxonomy.predecessors(n)]
            return (sum(parent_xs) / len(parent_xs) if parent_xs else 0, n)
        for i, n in enumerate(sorted(layer, key=order_key)):
            positions[n] = (i - (len(layer) - 1) / 2, -depth)
    return positions


def _draw_panel(ax, graph, positions, taxonomy_roots=(), show_edge_labels=True):
    """Use text-box boundaries, so arrowheads cannot disappear under labels."""
    texts = {}
    wrap_width = 11 if len(graph) <= 8 else 17
    for n in graph:
        attrs = graph.nodes[n]
        label = textwrap.fill(attrs.get("label", n).lower(), width=wrap_width)
        texts[n] = ax.text(
            *positions[n], f"{n}\n{label}", ha="center", va="center", fontsize=8,
            zorder=3,
            bbox={
                "boxstyle": "round,pad=0.4", "linewidth": 1.5 if n in taxonomy_roots else .8,
                "facecolor": TARGET_COLOR if attrs.get("is_target") == "TRUE" else AUXILIARY_COLOR,
                "edgecolor": ROOT_COLOR if n in taxonomy_roots else "#71808b",
            },
        )
    xs, ys = zip(*positions.values())
    ax.set_xlim(min(xs) - .75, max(xs) + .75)
    ax.set_ylim(min(ys) - .55, max(ys) + .6)
    ax.axis("off")
    ax.figure.canvas.draw()
    edge_rows = list(graph.edges(data=True))
    for u, v, attrs in edge_rows:
        relation = attrs.get("relation", "is_a")
        color = RELATION_COLORS.get(relation, "#52666b")
        arrow = FancyArrowPatch(
            positions[u], positions[v],
            patchA=texts[u].get_bbox_patch(), patchB=texts[v].get_bbox_patch(),
            arrowstyle="-|>", mutation_scale=12, linewidth=1.15,
            color=color, shrinkA=3, shrinkB=3, zorder=1,
        )
        ax.add_patch(arrow)
        if show_edge_labels:
            x = .48 * positions[u][0] + .52 * positions[v][0]
            y = .48 * positions[u][1] + .52 * positions[v][1]
            ax.text(
                x, y, relation, fontsize=6 if relation == "is_a" else 9,
                color=color, ha="center", va="center", zorder=2,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": .3, "alpha": .9},
            )


def draw_taxonomy(taxonomy):
    """One large main component, seven small components, and ALL isolates."""
    components = sorted(
        (c for c in nx.weakly_connected_components(taxonomy) if len(c) > 1),
        key=lambda c: (-len(c), tuple(sorted(c))),
    )
    isolates = sorted(nx.isolates(taxonomy))
    small_rows = math.ceil(max(len(components) - 1, 0) / 4)
    heights = ([3.0] if components else []) + [1.6] * small_rows + ([.9] if isolates else [])
    fig = plt.figure(figsize=(25, 4 * sum(heights)), layout="constrained")
    grid = fig.add_gridspec(len(heights), 4, height_ratios=heights)
    row_offset = 0
    for i, comp in enumerate(components):
        if i == 0:
            ax = fig.add_subplot(grid[0, :])
            row_offset = 1
        else:
            ax = fig.add_subplot(grid[row_offset + (i - 1) // 4, (i - 1) % 4])
        graph = taxonomy.subgraph(comp).copy()
        roots = [n for n in graph if graph.in_degree(n) == 0]
        _draw_panel(ax, graph, layered_positions(graph), roots)
        ax.set_title(f"K{i + 1:02d} · {len(comp)} узлов · корней в снимке: {len(roots)}", fontsize=11)
    if isolates:
        ax = fig.add_subplot(grid[len(heights) - 1, :])
        graph = taxonomy.subgraph(isolates).copy()
        _draw_panel(ax, graph, {n: (i, 0) for i, n in enumerate(isolates)}, isolates)
        ax.set_title("Изолированные узлы снимка: все сохранены, гиперонимы не добавлены", fontsize=11)
    fig.suptitle(
        f"HA2 · таксономия: {len(taxonomy)} узлов, {taxonomy.number_of_edges()} рёбер\n"
        "Все стрелки is_a: родитель → потомок. Вертикальный ранг = максимальная глубина.",
        fontsize=17,
    )
    fig.legend(handles=[
        Patch(facecolor=TARGET_COLOR, edgecolor="#71808b", label="Целевой узел"),
        Patch(facecolor=AUXILIARY_COLOR, edgecolor="#71808b", label="Вспомогательный узел"),
        Patch(facecolor="white", edgecolor=ROOT_COLOR, label="Корень только в текущем снимке"),
        Line2D([0], [0], color=RELATION_COLORS["is_a"], label="is_a: родитель → потомок"),
    ], loc="outside lower center", ncols=4, frameon=False)
    return fig


def draw_non_taxonomic(full):
    """Separate labelled edge panels avoid mixing category depth with functions."""
    edges = [(u, v, d) for u, v, d in full.edges(data=True) if d["relation"] != "is_a"]
    fig, axes = plt.subplots(
        max(len(edges), 1), 1, figsize=(14, max(3, len(edges) * 2.9)),
        squeeze=False, layout="constrained",
    )
    incident = set()
    for ax, (u, v, attrs) in zip(axes[:, 0], edges):
        graph = nx.DiGraph()
        graph.add_nodes_from((n, dict(full.nodes[n])) for n in (u, v))
        graph.add_edge(u, v, **attrs)
        _draw_panel(ax, graph, {u: (0, 0), v: (2, 0)})
        ax.set_title(
            f'{attrs["justification"]}\n'
            f'provenance={attrs["provenance"]}; confidence={attrs["confidence"]}',
            fontsize=10,
        )
        incident.update((u, v))
    if not edges:
        axes[0, 0].text(.5, .5, "Нет нетаксономических рёбер", ha="center")
        axes[0, 0].axis("off")
    fig.suptitle(
        "HA2 · нетаксономические связи (не участвуют в расчётах)\n"
        f"Показаны {len(incident)} узлов; ещё {len(full) - len(incident)} не имеют связей этого слоя.",
        fontsize=14,
    )
    fig.legend(handles=[
        Line2D([0], [0], color=RELATION_COLORS.get(r, "#52666b"), label=r)
        for r in sorted({d["relation"] for _, _, d in edges})
    ], loc="outside lower center", ncols=3, frameon=False)
    return fig


def save_figures(data_dir, taxonomy_figure, relations_figure):
    output = Path(data_dir) / "figures"
    output.mkdir(exist_ok=True)
    for name, fig in (("taxonomy", taxonomy_figure), ("non_taxonomic", relations_figure)):
        for suffix in ("png", "svg"):
            with plt.rc_context({"svg.hashsalt": "ha2-role3"}):
                fig.savefig(output / f"{name}.{suffix}", dpi=160, bbox_inches="tight",
                            metadata={"Date": None} if suffix == "svg" else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    nodes, edges = load_inputs(args.data_dir)
    full, taxonomy = build_graphs(nodes, edges)
    scores, constants = compute_specificity(taxonomy)
    validate_scores(taxonomy, scores)
    audit = audit_directions(args.data_dir, nodes, edges)
    if not audit.source_direction_confirmed.all():
        raise ValueError("Some source directions are not confirmed; see audit_directions")
    checks = export_results(
        args.data_dir, nodes, edges, full, taxonomy, scores, constants, audit
    )
    taxonomy_figure, relations_figure = draw_taxonomy(taxonomy), draw_non_taxonomic(full)
    save_figures(args.data_dir, taxonomy_figure, relations_figure)
    plt.close(taxonomy_figure)
    plt.close(relations_figure)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
