"""Coverage is separate from HA2 specificity and from graph perturbation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from ha2_coverage import (
    ROOT, FEATURES, QUADRANTS, baseline_identity, counts, normalization, quadrant_coverage, summary,
)
from ha2_perturbation import quadrant_coverage as legacy_quadrant_coverage


def example():
    # Raw C = [0,0,4,4,2], raw S = [1,5,1,5,3].
    concepts, scores = [], []
    for node, c, depth in zip("abcde", [0, 0, 4, 4, 2], [1, 5, 1, 5, 3]):
        row = dict(concept_id=node, term=node, **{f: 0 for f in FEATURES})
        row["visual"] = min(c, 2)
        row["motor"] = max(c - 2, 0)
        concepts.append(row)
        scores.append(dict(
            node_id=node, label=node, is_target="TRUE", depth_min=depth,
            taxonomy_isolate="False",
        ))
    # An auxiliary root/branch must not change target-only min/max.
    scores.append(dict(
        node_id="aux", label="aux", is_target="FALSE", depth_min=99,
        taxonomy_isolate=False,
    ))
    return pd.DataFrame(concepts), pd.DataFrame(scores)


class CoverageTests(unittest.TestCase):
    def test_target_only_normalization_and_equality_at_threshold(self):
        concepts, scores = example()
        original = concepts.copy(deep=True)
        table = quadrant_coverage(scores, concepts)
        self.assertEqual(table.quadrant.to_dict(), {
            "a": "C- S-", "b": "C- S+", "c": "C+ S-", "d": "C+ S+", "e": "C+ S+",
        })
        self.assertEqual((table.loc["e", "C_norm"], table.loc["e", "S_norm"]), (.5, .5))
        self.assertEqual(normalization(table)["S_max"], 5)
        self.assertEqual(counts(table).loc[QUADRANTS].tolist(), [2, 1, 1, 1])
        self.assertFalse(summary(table)["quadrant_minimum_met"])
        pd.testing.assert_frame_equal(original, concepts)

    def test_isolated_target_is_missing_not_low_specificity(self):
        concepts, scores = example()
        scores.loc[scores.node_id.eq("a"), "taxonomy_isolate"] = "True"
        table = quadrant_coverage(scores, concepts)
        self.assertTrue(pd.isna(table.loc["a", "S_raw"]))
        self.assertEqual(table.loc["a", "quadrant"], "no S")
        self.assertEqual(table.loc["a", "S_missing_reason"], "isolated_in_baseline")
        self.assertEqual(normalization(table)["usable_S_count"], 4)
        self.assertFalse(summary(table)["all_targets_have_S"])

    def test_invalid_features_identities_and_undefined_normalization_fail(self):
        for case in (
            "feature_out_of_scale", "missing_feature", "duplicate_id", "wrong_label",
            "invalid_isolate", "negative_depth", "constant_C", "constant_S",
        ):
            with self.subTest(case=case):
                concepts, scores = example()
                if case == "feature_out_of_scale":
                    concepts.loc[0, "visual"] = 3
                elif case == "missing_feature":
                    concepts = concepts.drop(columns="social")
                elif case == "duplicate_id":
                    concepts = pd.concat([concepts, concepts.iloc[:1]], ignore_index=True)
                elif case == "wrong_label":
                    scores.loc[0, "label"] = "different concept"
                elif case == "invalid_isolate":
                    scores.loc[0, "taxonomy_isolate"] = "unknown"
                elif case == "negative_depth":
                    scores.loc[0, "depth_min"] = -1
                elif case == "constant_C":
                    concepts[FEATURES] = 0
                elif case == "constant_S":
                    scores["depth_min"] = 1
                with self.assertRaises(ValueError):
                    quadrant_coverage(scores, concepts)

    def test_legacy_function_call_reads_the_revised_concept_table(self):
        scores = pd.read_csv(ROOT / "results/specificity.csv", dtype={"is_target": str})
        table = legacy_quadrant_coverage(scores)
        self.assertEqual(len(table), 25)
        self.assertEqual(table.loc["C22", "term"], "близость")
        self.assertEqual(table.loc["C22", "C_norm"], .5)
        self.assertEqual(counts(table).loc[QUADRANTS].tolist(), [5, 6, 7, 7])

    def test_cli_reproduces_current_counts_and_does_not_recalculate_baseline(self):
        with tempfile.TemporaryDirectory(prefix="ha1-coverage-") as tmp:
            directory = Path(tmp)
            source = json.loads((ROOT / "results/run_manifest.json").read_text())
            for name in source["input_sha256"]:
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            (directory / "results").mkdir(exist_ok=True)
            for name in [*source["baseline_output_sha256"], "run_manifest.json"]:
                shutil.copyfile(ROOT / "results" / name, directory / "results" / name)
            command = [sys.executable, str(ROOT / "ha2_coverage.py"), "--data-dir", tmp]
            run = subprocess.run(command, check=True, capture_output=True, text=True)
            report = json.loads(run.stdout)
            self.assertEqual(report["as_provided"]["quadrant_counts"],
                             dict(zip(QUADRANTS, [5, 6, 7, 7])))
            self.assertEqual(report["closeness_social_0"]["quadrant_counts"],
                             dict(zip(QUADRANTS, [5, 5, 7, 8])))
            self.assertTrue(report["as_provided"]["quadrant_minimum_met"])
            self.assertTrue(report["closeness_social_0"]["quadrant_minimum_met"])
            self.assertEqual(report["sensitivity_changed_concepts"], ["близость"])
            self.assertEqual(report["as_provided"]["usable_S_count"], 25)
            baseline = json.loads((directory / "results/run_manifest.json").read_text())
            self.assertEqual(baseline["coverage_status"], "computed")
            self.assertEqual(baseline["perturbation_status"], source["perturbation_status"])
            self.assertEqual(baseline["perturbation_performed"], source["perturbation_performed"])
            self.assertEqual(baseline_identity(source), baseline_identity(baseline))
            self.assertFalse((directory / "results/role4").exists())
            for name, expected in source["input_sha256"].items():
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected)
            for name, expected in source["baseline_output_sha256"].items():
                self.assertEqual(
                    hashlib.sha256((directory / "results" / name).read_bytes()).hexdigest(), expected
                )
            out = directory / "results/ha1_coverage"
            files = {p.name: p.read_bytes() for p in out.iterdir()}
            manifest = json.loads(files["run_manifest.json"])
            for name, digest in manifest["output_sha256"].items():
                self.assertEqual(hashlib.sha256(files[name]).hexdigest(), digest)
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(files, {p.name: p.read_bytes() for p in out.iterdir()})
            # Even a numerically harmless edit of the baseline must invalidate its fingerprint.
            saved = directory / "results/specificity.csv"
            saved.write_bytes(saved.read_bytes() + b"\n")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Stale baseline dependency", failed.stderr)
            self.assertEqual(files, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
