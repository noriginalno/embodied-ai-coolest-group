"""Small offline regressions; expected values follow hand-drawn DAGs."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import networkx as nx
import pandas as pd
import sync as input_sync

from ha2_analysis import (
    DATA_DIR, audit_directions, build_graphs, compute_specificity,
    export_results, layered_positions, load_inputs, validate_scores,
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
    def test_revised_exports_preserve_nodes_and_all_typed_edges(self):
        files = {
            "nodes": "Nodes.csv", "Edges": "Edges.csv",
            "Concepts_hw2": "Concepts_hw2.csv", "Sources": "Sources.csv",
        }
        raw = {
            name: input_sync.parse(name, (DATA_DIR / "raw" / filename).read_bytes())
            for name, filename in files.items()
        }
        nodes, edges, excluded = input_sync.build(raw)
        self.assertEqual(
            {r["node_id"]: r["label"] for r in nodes},
            {r["node_id"]: r["label"] for r in raw["nodes"]},
        )
        self.assertEqual(
            {(r["source"], r["relation"], r["target"]) for r in edges},
            {(r["source"], r["relation"], r["target"]) for r in raw["Edges"]},
        )
        self.assertEqual(excluded, [])
        self.assertEqual(next(r["label"] for r in nodes if r["node_id"] == "A01"),
                         "ПОСТОЯННАЯ СУЩНОСТЬ")
        self.assertEqual(next(r["label"] for r in nodes if r["node_id"] == "C10"), "ширина")
        expected_provenance = {
            "RuWordNet": "wordnet", "RuWordNet extraction": "wordnet",
            "RuWordNet/HA1": "wordnet", "RuWordNet page": "wordnet",
            "manual": "manual", "RuWordNet gloss/manual": "manual",
            "RuWordNet association/manual": "manual",
        }
        for prepared, source in zip(edges, raw["Edges"]):
            self.assertEqual(prepared["source_provenance"], source["provenance"])
            self.assertEqual(prepared["provenance"], expected_provenance[source["provenance"]])
            for key in ("source", "relation", "target", "confidence", "justification"):
                self.assertEqual(prepared[key], source[key])

    def test_provenance_mapping_is_explicit_and_preserves_manual_derivations(self):
        for source in ("seed", "manual", "wordnet", "framenet"):
            with self.subTest(source=source):
                self.assertEqual(input_sync.canonical_provenance(source), source)
        self.assertEqual(input_sync.canonical_provenance(" RuWordNet page "), "wordnet")
        self.assertEqual(input_sync.canonical_provenance("RuWordNet gloss/manual"), "manual")
        self.assertEqual(input_sync.canonical_provenance("RuWordNet association/manual"), "manual")
        with self.assertRaisesRegex(ValueError, "Unsupported edge provenance"):
            input_sync.canonical_provenance("unverified WordNet guess")

    def test_bom_and_revised_edge_schema_are_read_without_reversing_edges(self):
        content = (
            "\ufeffsource,relation,target,provenance,confidence,justification\r\n"
            "parent,is_a,child,RuWordNet,medium,Parent to child\r\n"
        ).encode("utf-8")
        self.assertEqual(input_sync.parse("Edges", content), [{
            "source": "parent", "relation": "is_a", "target": "child",
            "provenance": "RuWordNet", "confidence": "medium",
            "justification": "Parent to child",
        }])
        with self.assertRaisesRegex(ValueError, "Unexpected columns"):
            input_sync.parse("Edges", b"parent,child,relation,provenance\np,c,is_a,wordnet\n")

    def test_pos_requires_identity_bound_annotations_and_does_not_infer_from_ili(self):
        nodes = [
            dict(node_id="C10", label="ширина", source_id="i63979"),
            dict(node_id="n", label="предмет", source_id="123-N; i100"),
            dict(node_id="v", label="двигаться", source_id="456-V"),
            dict(node_id="new", label="новый термин", source_id="i999"),
        ]
        nodes = [dict(node_type="target", is_target="TRUE", source="test", **n) for n in nodes]
        annotation = dict(
            node_id="C10", label="ширина", source_id="i63979", POS="NOUN",
            provenance="manual", justification="Lexical annotation, not an ILI lookup",
        )
        prepared = {r["node_id"]: r for r in input_sync.annotate_pos(nodes, [annotation])}
        self.assertEqual(prepared["C10"]["POS"], "NOUN")
        self.assertEqual(prepared["C10"]["POS_provenance"], "manual:pos_annotations.csv")
        self.assertEqual(prepared["n"]["POS"], "NOUN")
        self.assertEqual(prepared["n"]["POS_provenance"], "wordnet:source_id_suffix")
        self.assertEqual(prepared["v"]["POS"], "VERB")
        self.assertEqual(prepared["new"]["POS"], "UNKNOWN")
        self.assertEqual(input_sync.annotate_pos(nodes, [])[0]["POS"], "UNKNOWN")
        for changes, message in [
            ({"label": "между"}, "Stale POS"),
            ({"source_id": "i000"}, "Stale POS"),
            ({"node_id": "absent"}, "absent nodes"),
            ({"POS": "inferred_noun"}, "Invalid manual POS"),
            ({"provenance": "wordnet"}, "Invalid manual POS"),
        ]:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    input_sync.annotate_pos(nodes, [dict(annotation, **changes)])
        conflict = dict(
            annotation, node_id="n", label="предмет", source_id="123-N; i100", POS="VERB"
        )
        with self.assertRaisesRegex(ValueError, "conflicts with synset suffix"):
            input_sync.annotate_pos(nodes, [conflict])
        with self.assertRaisesRegex(ValueError, "Conflicting POS suffixes"):
            input_sync.annotate_pos([dict(nodes[1], source_id="123-N; 456-V")], [])

    def test_import_cli_is_idempotent_checks_staleness_and_leaves_raw_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="ha2-inputs-") as tmp:
            directory = Path(tmp)
            (directory / "raw").mkdir()
            for name in input_sync.RAW_FILES.values():
                shutil.copyfile(DATA_DIR / "raw" / name, directory / "raw" / name)
            shutil.copyfile(DATA_DIR / "pos_annotations.csv", directory / "pos_annotations.csv")
            raw_before = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (directory / "raw").iterdir()
            }
            command = [sys.executable, str(DATA_DIR / "sync.py"), "--data-dir", tmp]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads(first.stdout)
            self.assertEqual((summary["node_count"], summary["target_count"], summary["edge_count"]),
                             (44, 25, 41))
            self.assertEqual(summary["provenance_counts"], {"wordnet": 37, "manual": 4})
            self.assertEqual(summary["POS_counts"], {"NOUN": 44})
            nodes, edges = load_inputs(directory)
            self.assertEqual(len(nodes), 44)
            self.assertEqual(len(edges), 41)
            self.assertEqual(
                nodes.POS_provenance.value_counts().to_dict(),
                {"manual:pos_annotations.csv": 28, "wordnet:source_id_suffix": 16},
            )
            manifest = json.loads((directory / "prepared_inputs.json").read_text())
            for name, digest in manifest["prepared_sha256"].items():
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest)
            written = {
                name: (directory / name).read_bytes()
                for name in [*input_sync.OUTPUT_FIELDS, "prepared_inputs.json"]
            }
            refresh = subprocess.run(
                command + ["--refresh"], capture_output=True, text=True
            )
            self.assertEqual(refresh.returncode, 2)
            self.assertIn("--refresh is disabled", refresh.stderr)
            self.assertEqual(written, {name: (directory / name).read_bytes() for name in written})
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(written, {name: (directory / name).read_bytes() for name in written})
            subprocess.run(command + ["--check"], check=True, capture_output=True, text=True)
            self.assertEqual(raw_before, {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (directory / "raw").iterdir()
            })
            # A reused ID must not silently acquire a new concept's definition/POS.
            node_file = directory / "nodes.csv"
            node_file.write_text(node_file.read_text().replace("C10,ширина,", "C10,между,", 1))
            with self.assertRaisesRegex(ValueError, "nodes.csv is stale"):
                load_inputs(directory)
            input_sync.prepare(directory)
            # Invalid raw provenance must fail before overwriting a previous valid export.
            edge_file = directory / "raw/Edges.csv"
            edge_file.write_bytes(edge_file.read_bytes().replace(b",RuWordNet,", b",unknown,", 1))
            before_invalid = {name: (directory / name).read_bytes() for name in written}
            with self.assertRaisesRegex(ValueError, "Unsupported edge provenance"):
                input_sync.prepare(directory)
            self.assertEqual(before_invalid, {
                name: (directory / name).read_bytes() for name in written
            })

    def test_baseline_export_records_actual_snapshot_and_pending_role4(self):
        with tempfile.TemporaryDirectory(prefix="ha2-baseline-") as tmp:
            directory = Path(tmp)
            (directory / "raw").mkdir()
            for name in input_sync.RAW_FILES.values():
                shutil.copyfile(DATA_DIR / "raw" / name, directory / "raw" / name)
            shutil.copyfile(DATA_DIR / "pos_annotations.csv", directory / "pos_annotations.csv")
            prepared = input_sync.prepare(directory)
            nodes, edges = load_inputs(directory)
            full, taxonomy = build_graphs(nodes, edges)
            scores, constants = compute_specificity(taxonomy)
            audit = audit_directions(directory, nodes, edges)
            checks = export_results(
                directory, nodes, edges, full, taxonomy, scores, constants, audit
            )
            self.assertEqual(checks["source_commit"], prepared["source_snapshot_commit"])
            self.assertEqual(checks["POS_counts"], {"NOUN": 44})
            self.assertEqual(checks["multiple_parent_count"], 7)
            self.assertEqual(checks["unequal_depth_nodes"], [])
            self.assertNotIn("56 nodes", " ".join(checks["warnings"]))
            self.assertNotIn("ADP and ADV", " ".join(checks["warnings"]))
            manifest = json.loads((directory / "results/run_manifest.json").read_text())
            self.assertEqual(manifest["run_kind"], "baseline")
            self.assertEqual(manifest["coverage_status"], "not_recomputed")
            self.assertEqual(manifest["perturbation_status"], "not_recomputed")
            self.assertIn("raw/Nodes.csv", manifest["input_sha256"])
            self.assertIn("raw/Sources.csv", manifest["input_sha256"])
            self.assertIn("ha2_analysis.py", manifest["code_sha256"])
            self.assertFalse((directory / "results/role4").exists())
            for name, digest in manifest["baseline_output_sha256"].items():
                self.assertEqual(
                    hashlib.sha256((directory / "results" / name).read_bytes()).hexdigest(), digest
                )

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
        nodes, edges = tables(["a", "b"], [("a", "is_a", "b")])
        edges.loc[0, "provenance"] = "RuWordNet"
        with self.assertRaisesRegex(ValueError, "Unsupported edge provenance"):
            build_graphs(nodes, edges)

    def test_real_input_snapshot_is_covered_without_modifying_csvs(self):
        paths = [DATA_DIR / name for name in ("nodes.csv", "edges.csv")]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        nodes, edges = load_inputs()
        full, taxonomy = build_graphs(nodes, edges)
        audit = audit_directions(DATA_DIR, nodes, edges)
        self.assertTrue(audit.source_direction_confirmed.all())
        self.assertEqual(full.number_of_edges(), len(edges))
        self.assertEqual(set(taxonomy), set(nodes.node_id))
        self.assertEqual(
            set(taxonomy.edges),
            set(edges.loc[edges.relation.eq("is_a"), ["source", "target"]].itertuples(index=False, name=None)),
        )
        self.assertEqual((len(nodes), len(edges), taxonomy.number_of_edges()), (44, 41, 37))
        self.assertEqual(nodes.set_index("node_id").loc["C10", "POS"], "NOUN")
        self.assertEqual(nodes.set_index("node_id").loc["C18", "label"], "центр")
        self.assertEqual(nodes.set_index("node_id").loc["C25", "label"], "ориентация")
        self.assertEqual(set(nodes.POS), {"NOUN"})
        scores, constants = compute_specificity(taxonomy)
        validate_scores(taxonomy, scores)
        self.assertEqual(set(scores.node_id), set(nodes.node_id))
        by_id = scores.set_index("node_id")
        # A01 -> A02 -> A03 -> A04 -> distance -> height.
        self.assertEqual(by_id.loc["C05", "depth_min"], 5)
        # A01 -> physical entity -> place -> point -> center.
        self.assertEqual(by_id.loc["C18", "depth_min"], 4)
        self.assertEqual(by_id.loc["A01", "descendants"], 18)
        self.assertTrue(scores.depth_min.eq(scores.depth_max).all())
        self.assertTrue(scores.S_depth.eq(scores.S_depth_pos).all())
        self.assertTrue(scores.S_breadth.eq(scores.S_breadth_pos).all())
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__ == "__main__":
    unittest.main()
