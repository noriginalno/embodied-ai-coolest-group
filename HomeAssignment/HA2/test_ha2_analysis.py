"""Small offline regressions; expected values follow hand-drawn DAGs."""
import hashlib
import math
import unittest

import networkx as nx
import pandas as pd

from ha2_analysis import (
    DATA_DIR, audit_directions, build_graphs, compute_specificity,
    layered_positions, load_inputs, validate_scores,
)


def tables(ids, edge_rows, pos=None):
    nodes = pd.DataFrame([
        dict(node_id=n, label=n, node_type="target", is_target="TRUE",
             POS=(pos or {}).get(n, "NOUN"))
        for n in ids
    ])
    edges = pd.DataFrame([
        dict(source=u, relation=r, target=v, provenance="manual",
             confidence="high", justification="Synthetic test edge")
        for u, r, v in edge_rows
    ], columns=["source", "relation", "target", "provenance", "confidence", "justification"])
    return nodes, edges


class SpecificityTests(unittest.TestCase):
    def test_multiple_roots_paths_and_distinct_descendants(self):
        # Two routes to c: r->a->b->c and r->c. Six isolates make N=10.
        nodes, edges = tables(
            ["r", "a", "b", "c"] + [f"i{i}" for i in range(6)],
            [(u, "is_a", v) for u, v in [("r", "a"), ("a", "b"), ("b", "c"), ("r", "c")]],
        )
        _, taxonomy = build_graphs(nodes, edges)
        scores, norms = compute_specificity(taxonomy)
        validate_scores(taxonomy, scores)
        s = scores.set_index("node_id")
        self.assertEqual((s.loc["c", "depth_min"], s.loc["c", "depth_max"]), (1, 3))
        self.assertEqual(s.loc["c", "root_path_count"], 2)
        self.assertEqual(s.loc["r", "descendants"], 3)  # distinct nodes, not paths
        self.assertEqual(s.loc["c", "S_depth"], .5)  # max shortest depth = 2
        self.assertAlmostEqual(s.loc["c", "S_depth_range_min"], 1 / 3)
        self.assertEqual(s.loc["c", "S_depth_range_max"], 1)
        self.assertEqual(s.loc["r", "S_breadth"], 0)
        self.assertAlmostEqual(s.loc["a", "S_breadth"], 1 - math.log(3) / math.log(4))
        self.assertEqual(s.loc["c", "S_breadth"], 1)
        self.assertTrue(s.loc["i0", "taxonomy_isolate"])
        self.assertEqual(set(s.index), set(nodes.node_id))
        self.assertEqual(norms.set_index("scope").loc["global", "max_depth_longest"], 3)
        positions = layered_positions(taxonomy)
        self.assertTrue(all(positions[u][1] > positions[v][1] for u, v in taxonomy.edges))

    def test_all_observed_roots_are_used_without_a_super_root(self):
        nodes, edges = tables(
            ["r1", "a", "b", "r2"] + [f"i{i}" for i in range(6)],
            [("r1", "is_a", "a"), ("a", "is_a", "b"), ("r2", "is_a", "b")],
        )
        _, taxonomy = build_graphs(nodes, edges)
        scores, _ = compute_specificity(taxonomy)
        b = scores.set_index("node_id").loc["b"]
        self.assertEqual((b.depth_min, b.depth_max), (1, 2))
        self.assertEqual(b.root_ids, "r1;r2")
        self.assertEqual(b.root_count, 2)
        self.assertEqual(len(taxonomy), 10)

    def test_parallel_types_and_full_network_cycle_do_not_change_scores(self):
        ids = ["r", "a"] + [f"i{i}" for i in range(8)]
        nodes, baseline_edges = tables(ids, [("r", "is_a", "a")])
        _, typed_edges = tables(ids, [
            ("r", "is_a", "a"), ("r", "associated_with", "a"),
            ("a", "has_endpoint", "r"),
        ])
        _, baseline = build_graphs(nodes, baseline_edges)
        full, typed_taxonomy = build_graphs(nodes, typed_edges)
        self.assertEqual(full.number_of_edges("r", "a"), 2)
        self.assertFalse(nx.is_directed_acyclic_graph(full))
        self.assertTrue(nx.is_directed_acyclic_graph(typed_taxonomy))
        pd.testing.assert_frame_equal(
            compute_specificity(baseline)[0], compute_specificity(typed_taxonomy)[0]
        )

    def test_small_or_zero_information_graphs_keep_raw_values_without_fake_scores(self):
        for count, links in [
            (2, [("i0", "is_a", "i1")]), (10, []),
        ]:
            with self.subTest(count=count):
                nodes, edges = tables([f"i{i}" for i in range(count)], links)
                _, taxonomy = build_graphs(nodes, edges)
                scores, _ = compute_specificity(taxonomy)
                self.assertEqual(len(scores), count)
                self.assertTrue(scores.S_depth.isna().all())
                self.assertTrue(scores.S_breadth.isna().all())
                if count == 2:
                    self.assertEqual(scores.set_index("node_id").loc["i1", "depth_min"], 1)

    def test_pos_uses_induced_graph_and_alternative_formula_is_kept_separate(self):
        nodes, edges = tables(
            ["r", "a", "v", "w"] + [f"i{i}" for i in range(6)],
            [("r", "is_a", "a"), ("a", "is_a", "v"), ("v", "is_a", "w")],
            pos={"v": "VERB", "w": "VERB"},
        )
        _, taxonomy = build_graphs(nodes, edges)
        # Explicitly opt in to tiny synthetic-graph normalization for this arithmetic check.
        scores, constants = compute_specificity(taxonomy, min_normalization_nodes=1)
        s = scores.set_index("node_id")
        self.assertEqual(s.loc["v", "depth_min"], 2)
        self.assertEqual(s.loc["v", "pos_depth_min"], 0)
        self.assertEqual(s.loc["w", "pos_depth_min"], 1)
        self.assertEqual(s.loc["w", "S_depth_pos"], 1)
        self.assertAlmostEqual(s.loc["w", "slide28_raw_min"], 1 + math.log(1 / 2))
        self.assertEqual(s.loc["v", "S_breadth_pos"], 0)
        self.assertAlmostEqual(s.loc["v", "slide28_raw_min"], 0)
        self.assertEqual(constants.set_index("scope").loc["POS:VERB", "node_count"], 2)

    def test_invalid_inputs_fail_instead_of_silently_changing_graph(self):
        cases = [
            (["a", "a"], [], "unique"),
            (["a"], [("a", "is_a", "missing")], "Unknown edge"),
            (["a", "b"], [("a", "is_a", "b"), ("a", "is_a", "b")], "Duplicate typed"),
            (["a", "b"], [("a", "is_a", "b"), ("b", "is_a", "a")], "is_a cycle"),
            (["a"], [("a", "is_a", "a")], "is_a cycle"),
        ]
        for ids, rows, message in cases:
            with self.subTest(message=message, rows=rows):
                with self.assertRaisesRegex(ValueError, message):
                    build_graphs(*tables(ids, rows))

    def test_real_input_snapshot_is_covered_without_modifying_csvs(self):
        paths = [DATA_DIR / name for name in ("nodes.csv", "edges.csv")]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        nodes, edges = load_inputs()
        full, taxonomy = build_graphs(nodes, edges)
        scores, constants = compute_specificity(taxonomy)
        validate_scores(taxonomy, scores)
        audit = audit_directions(DATA_DIR, nodes, edges)
        self.assertTrue(audit.source_direction_confirmed.all())
        self.assertEqual(full.number_of_edges(), len(edges))
        self.assertEqual(set(scores.node_id), set(nodes.node_id))
        self.assertEqual(
            set(taxonomy.edges),
            set(edges.loc[edges.relation.eq("is_a"), ["source", "target"]].itertuples(index=False, name=None)),
        )
        s = scores.set_index("node_id")
        # HA1 root->abstract entity->property->quantity->distance->height.
        self.assertEqual(s.loc["C05", "depth_min"], 5)
        # Permanent entity -> physical entity -> place.
        self.assertEqual((s.loc["C13", "depth_min"], s.loc["C13", "depth_max"]), (2, 2))
        self.assertEqual(len(scores[scores.is_target.eq("TRUE")]), 25)
        self.assertEqual(constants.set_index("scope").loc["global", "node_count"], len(nodes))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__ == "__main__":
    unittest.main()
