"""Experiment lifecycle and an independent stdlib oracle over exported CSVs."""
from collections import deque
import copy
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import ha2_perturbation as experiment
from ha2_analysis import load_inputs


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify_export(output):
    """No NetworkX or production math: BFS, explicit paths and reachability sets."""
    output = Path(output)
    nodes = {r["node_id"]: r for r in read_csv(output / "baseline_nodes.csv")}
    expected, constants, structures = {}, {}, {}
    for stage in ("baseline", "after_add", "after_remove"):
        edges = read_csv(output / f"{stage}_edges.csv")
        parents, children = ({n: set() for n in nodes} for _ in range(2))
        for e in edges:
            if e["relation"] == "is_a":
                children[e["source"]].add(e["target"])
                parents[e["target"]].add(e["source"])
        roots = {n for n in nodes if not parents[n]}
        depths = dict.fromkeys(roots, 0)
        queue = deque(sorted(roots))
        while queue:
            n = queue.popleft()
            for child in sorted(children[n]):
                if child not in depths:
                    depths[child] = depths[n] + 1
                    queue.append(child)
        assert set(depths) == set(nodes)

        def paths(n, seen=()):
            assert n not in seen, "Cycle in exported taxonomy"
            return ([p + (n,) for parent in parents[n] for p in paths(parent, seen + (n,))]
                    if parents[n] else [(n,)])

        all_paths = {n: paths(n) for n in nodes}
        longest = {n: max(len(p) - 1 for p in ps) for n, ps in all_paths.items()}
        descendants = {}
        for n in nodes:
            visited, pending = set(), list(children[n])
            while pending:
                v = pending.pop()
                if v not in visited:
                    visited.add(v)
                    pending.extend(children[v])
            descendants[n] = len(visited)
        components, remaining = [], set(nodes)
        while remaining:
            pending, comp = [min(remaining)], set()
            while pending:
                n = pending.pop()
                if n not in comp:
                    comp.add(n)
                    pending.extend((parents[n] | children[n]) - comp)
            components.append(comp)
            remaining -= comp
        components.sort(key=lambda c: (-len(c), tuple(sorted(c))))
        memberships = {n: (f"K{i:02d}", len(c)) for i, c in enumerate(components, 1) for n in c}
        ds, dl, maximum = max(depths.values()), max(longest.values()), max(descendants.values())
        denominator = math.log1p(maximum)
        constants[stage] = dict(Ds=ds, Dl=dl, maximum=maximum, denominator=denominator)
        scores = read_csv(output / f"{stage}_specificity.csv")
        assert len(scores) == len(nodes) and {r["node_id"] for r in scores} == set(nodes)
        values = {}
        for row in scores:
            n = row["node_id"]
            assert row["label"] == nodes[n]["label"] and row["is_target"] == nodes[n]["is_target"]
            assert int(row["depth_min"]) == depths[n] and int(row["depth_max"]) == longest[n]
            assert int(row["descendants"]) == descendants[n]
            assert int(row["root_path_count"]) == len(all_paths[n])
            assert int(row["parent_count"]) == len(parents[n])
            assert set(row["parent_ids"].split(";")) - {""} == parents[n]
            assert set(row["root_ids"].split(";")) == {p[0] for p in all_paths[n]}
            assert int(row["root_count"]) == len({p[0] for p in all_paths[n]})
            assert (row["component_id"], int(row["component_size"])) == memberships[n]
            assert (row["taxonomy_isolate"] == "True") == (not parents[n] and not children[n])
            values[n] = {
                "S_depth": depths[n] / ds,
                "S_breadth": 1 - math.log1p(descendants[n]) / denominator,
                "S_depth_range_min": depths[n] / dl, "S_depth_range_max": longest[n] / dl,
            }
            for column, value in values[n].items():
                assert math.isclose(float(row[column]), value, rel_tol=0, abs_tol=1e-12), (stage,n,column)
            values[n].update(depth_min=depths[n], descendants=descendants[n])
        norms = read_csv(output / f"{stage}_normalization.csv")
        assert {r["scope"] for r in norms} == {"global", "POS:NOUN"}
        for norm in norms:
            assert [int(norm[k]) for k in (
                "node_count", "max_depth_shortest", "max_depth_longest", "max_descendants"
            )] == [len(nodes), ds, dl, maximum]
            assert math.isclose(float(norm["breadth_log_denominator"]), denominator, abs_tol=1e-12)
        for suffix, selected in (("graph", edges), ("taxonomy_graph", [
            e for e in edges if e["relation"] == "is_a"
        ])):
            graph = json.loads((output / f"{stage}_{suffix}.json").read_text())
            assert len(graph["nodes"]) == len(nodes) and len(graph["edges"]) == len(selected)
            assert {r["id"] for r in graph["nodes"]} == set(nodes)
            assert {(e["source"], e["relation"], e["target"]) for e in graph["edges"]} == {
                (e["source"], e["relation"], e["target"]) for e in selected
            }
        expected[stage] = values
        structures[stage] = dict(
            node_count=len(nodes), typed_edges=len(edges),
            taxonomy_edges=sum(len(v) for v in children.values()), components=len(components),
            roots=len(roots), isolates=sum(not parents[n] and not children[n] for n in nodes),
            multiple_parents=sum(len(v) > 1 for v in parents.values()),
            unequal_depth_nodes=sum(depths[n] != longest[n] for n in nodes),
        )
    for row in read_csv(output / "structure.csv"):
        for column, value in structures[row["stage"]].items():
            assert int(row[column]) == value
    expected_keys, large_keys = set(), set()
    deltas = read_csv(output / "deltas.csv")
    assert len(deltas) == len(nodes) * 2 * 3
    for row in deltas:
        a, b, n, metric = row["from"], row["to"], row["node_id"], row["metric"]
        key = (a, b, n, metric)
        assert key not in expected_keys
        expected_keys.add(key)
        va, vb = expected[a][n], expected[b][n]
        delta = vb[metric] - va[metric]
        assert math.isclose(float(row["delta"]), delta, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(float(row["S_from"]), va[metric], abs_tol=1e-12)
        assert math.isclose(float(row["S_to"]), vb[metric], abs_tol=1e-12)
        raw = "depth_min" if metric == "S_depth" else "descendants"
        assert int(row["raw_from"]) == va[raw] and int(row["raw_to"]) == vb[raw]
        da = constants[a]["Ds" if metric == "S_depth" else "denominator"]
        db = constants[b]["Ds" if metric == "S_depth" else "denominator"]
        assert math.isclose(float(row["denominator_from"]), da, abs_tol=1e-12)
        assert math.isclose(float(row["denominator_to"]), db, abs_tol=1e-12)
        structural = float(row["delta_structure_from_scale"])
        normalization = float(row["delta_normalization"])
        assert math.isclose(structural + normalization, delta, rel_tol=0, abs_tol=1e-12)
        if va[raw] == vb[raw]:
            assert abs(structural) < 1e-12
        if da == db:
            assert abs(normalization) < 1e-12
        if metric == "S_depth":
            exact = Fraction(vb[raw], db) - Fraction(va[raw], da)
            large = abs(exact) > Fraction(1, 10)
        else:
            large = abs(delta) > .1 + 1e-12
        assert (row["exceeds_threshold"] == "True") == large
        if large:
            large_keys.add(key)
    assert expected_keys == {
        (a, b, n, metric) for a, b in (
            ("baseline","after_add"), ("baseline","after_remove"), ("after_add","after_remove")
        ) for n in nodes for metric in ("S_depth","S_breadth")
    }
    selected = read_csv(output / "changes_gt_0_1.csv")
    assert len(selected) == len(large_keys)
    assert {(r["from"],r["to"],r["node_id"],r["metric"]) for r in selected} == large_keys
    for row in read_csv(output / "change_summary.csv"):
        subset = [key for key in large_keys
                  if (key[0],key[1],key[3]) == (row["from"],row["to"],row["metric"])]
        assert int(row["all_nodes_exceeding"]) == len(subset)
        assert int(row["target_nodes_exceeding"]) == sum(
            nodes[key[2]]["is_target"] == "TRUE" for key in subset
        )
    scenario = json.loads((output / "scenario.json").read_text())
    keys_by_stage = {
        s: {(r["source"],r["relation"],r["target"])
            for r in read_csv(output / f"{s}_edges.csv")}
        for s in expected
    }
    adds, removes = (
        {(r["source"],r["relation"],r["target"]) for r in scenario[k]}
        for k in ("additions","removals")
    )
    assert len(adds) == len(removes) == 3 and removes <= keys_by_stage["baseline"]
    assert not adds & keys_by_stage["baseline"]
    assert keys_by_stage["after_add"] == keys_by_stage["baseline"] | adds
    assert keys_by_stage["after_remove"] == keys_by_stage["after_add"] - removes
    comparison = read_csv(output / "comparison.csv")
    assert len(comparison) == len(nodes)
    saved_scores = {
        stage: {r["node_id"]: r for r in read_csv(output / f"{stage}_specificity.csv")}
        for stage in expected
    }
    for row in comparison:
        n = row["node_id"]
        for stage, scores in expected.items():
            for column, value in saved_scores[stage][n].items():
                if f"{stage}_{column}" in row:
                    assert row[f"{stage}_{column}"] == value
            for metric in ("S_depth", "S_breadth"):
                assert math.isclose(float(row[f"{stage}_{metric}"]),scores[n][metric],abs_tol=1e-12)
        for a, b in (("baseline","after_add"), ("baseline","after_remove"), ("after_add","after_remove")):
            for metric in ("S_depth", "S_breadth"):
                assert math.isclose(
                    float(row[f"delta_{b}_vs_{a}_{metric}"]),
                    expected[b][n][metric] - expected[a][n][metric], rel_tol=0, abs_tol=1e-12,
                )
    return dict(verified_nodes=len(nodes), verified_states=3, verified_deltas=len(deltas),
                selected_changes=len(large_keys))


class PerturbationTests(unittest.TestCase):
    def test_strict_threshold_does_not_include_float_equality(self):
        for value in (0, .1, -.1, .10000000000000003):
            self.assertFalse(experiment.large_change(value))
        for value in (.100001, -.100001, 1):
            self.assertTrue(experiment.large_change(value))

    def test_scenario_preconditions_fail_instead_of_reusing_ids_or_invalid_edges(self):
        nodes, edges = load_inputs()
        wrong_identity = nodes.copy(deep=True)
        wrong_identity.loc[wrong_identity.node_id.eq("C14"), "label"] = "old concept"
        with self.assertRaisesRegex(ValueError, "identity C14"):
            experiment.prepare_states(wrong_identity, edges)
        for case, replacement, message in (
            ("duplicate", dict(source="A04", relation="is_a", target="A08"), "three distinct"),
            ("present", dict(source="A05", relation="is_a", target="C12"), "absent from baseline"),
            ("unknown", dict(source="missing", relation="is_a", target="C14"), "endpoints"),
            ("cycle", dict(source="C14", relation="is_a", target="A05"), "is_a cycle"),
        ):
            with self.subTest(case=case):
                additions = copy.deepcopy(experiment.ADDITIONS)
                additions[1].update(replacement)
                with patch.object(experiment, "ADDITIONS", additions):
                    with self.assertRaisesRegex(ValueError, message):
                        experiment.prepare_states(nodes, edges)
        removals = copy.deepcopy(experiment.REMOVALS)
        removals[0]["target"] = "C23"
        with patch.object(experiment, "REMOVALS", removals):
            with self.assertRaisesRegex(ValueError, "original baseline edges"):
                experiment.prepare_states(nodes, edges)

    def test_three_additions_then_three_original_removals_preserve_inputs_and_control(self):
        nodes, edges = load_inputs()
        before = edges.copy(deep=True)
        stages, checks, added = experiment.prepare_states(nodes, edges)
        pd.testing.assert_frame_equal(before, edges)
        self.assertTrue(all(c["passed"] for c in checks))
        self.assertTrue(all(r["provenance"] == "manual" for r in added))
        base = experiment.edge_keys(edges)
        a, b = (experiment.edge_keys(stages[k]["edges"]) for k in ("after_add", "after_remove"))
        self.assertEqual(a - base, {("A04","is_a","A08"),("A05","is_a","C14"),("C23","estimates","C04")})
        self.assertEqual(a - b, {("C13","is_a","C01"),("C14","is_a","C16"),("C04","is_a","C05")})
        self.assertEqual(len(base - b), 3)  # Not three removals independently from baseline.
        s = stages["after_add"]["scores"].set_index("node_id")
        self.assertEqual((s.loc["C14","depth_min"],s.loc["C14","depth_max"]), (2,3))
        final = stages["after_remove"]["scores"].set_index("node_id")
        self.assertEqual(set(final.index[final.taxonomy_isolate]), {"C05","C16"})
        self.assertEqual(final.loc["C18","depth_min"], 1)

    def test_cli_exports_verified_states_freezes_baseline_and_preserves_other_steps(self):
        with tempfile.TemporaryDirectory(prefix="ha2-perturbation-") as tmp:
            data = Path(tmp)
            source = json.loads((experiment.ROOT / "results/run_manifest.json").read_text())
            for name in source["input_sha256"]:
                target = data / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(experiment.ROOT / name, target)
            results = data / "results"
            results.mkdir(exist_ok=True)
            for name in [*source["baseline_output_sha256"], "run_manifest.json"]:
                shutil.copyfile(experiment.ROOT / "results" / name, results / name)
            # Other stages must not be overwritten, even when they already exist.
            for name in ("role4", "ha1_coverage"):
                (results / name).mkdir()
                (results / name / "keep.txt").write_text("protected")
            protected = [p for p in data.rglob("*") if p.is_file() and p.name != "run_manifest.json"]
            before = {p: p.read_bytes() for p in protected}
            command = [sys.executable, str(experiment.ROOT / "ha2_perturbation.py"), "--data-dir", tmp]
            run = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertFalse(run.stderr)
            self.assertTrue(json.loads(run.stdout)["baseline_preserved"])
            output = results / "perturbation"
            evidence = verify_export(output)
            self.assertEqual(evidence["verified_deltas"], 264)
            self.assertEqual(before, {p: p.read_bytes() for p in protected})
            manifest = json.loads((output / "run_manifest.json").read_text())
            for name, digest in manifest["output_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(),digest)
            for name, original in (
                ("baseline_edges.csv",data/"edges.csv"),
                ("baseline_graph.json",results/"typed_graph.json"),
                ("baseline_specificity.csv",results/"specificity.csv"),
            ):
                self.assertEqual((output/name).read_bytes(), original.read_bytes())
            final = json.loads((results / "run_manifest.json").read_text())
            self.assertEqual(final["coverage_status"],source["coverage_status"])
            self.assertEqual(final.get("coverage_manifest"),source.get("coverage_manifest"))
            self.assertEqual(final["perturbation_status"],"computed")
            self.assertTrue(final["perturbation_performed"])
            self.assertEqual(final["interpretation_written"], source["interpretation_written"])
            before_export_failure = {p.name: p.read_bytes() for p in output.iterdir()}
            # A scenario failure must leave the previous valid export intact.
            bad_additions = copy.deepcopy(experiment.ADDITIONS)
            bad_additions[1].update(source="C14", relation="is_a", target="A05")
            with patch.object(experiment, "ADDITIONS", bad_additions):
                with self.assertRaisesRegex(ValueError, "is_a cycle"):
                    experiment.run_perturbation(data)
            self.assertEqual(before_export_failure,{p.name: p.read_bytes() for p in output.iterdir()})
            written = {p.name: p.read_bytes() for p in output.iterdir()}
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(written,{p.name: p.read_bytes() for p in output.iterdir()})
            # Stale numerical input must fail without replacing valid results.
            (results/"specificity.csv").write_bytes((results/"specificity.csv").read_bytes()+b"\n")
            failed = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(failed.returncode,1)
            self.assertIn("Stale baseline dependency",failed.stderr)
            self.assertEqual(written,{p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
