"""Full artifact lifecycle: the CLI, figure inventory and stale-file detection."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np

import ha2_artifacts as artifacts
import sync as input_sync
from test_ha2_perturbation import verify_export


class ArtifactTests(unittest.TestCase):
    def test_long_taxonomy_edges_do_not_cross_unrelated_node_boxes_after_layout(self):
        def agg_figure(**kwargs):
            figure = Figure(**kwargs)
            FigureCanvasAgg(figure)
            return figure

        for stage in ("after_add", "after_remove"):
            with self.subTest(stage=stage):
                graph_path = artifacts.ROOT / f"results/perturbation/{stage}_taxonomy_graph.json"
                taxonomy = nx.node_link_graph(json.loads(graph_path.read_text()))
                # Standalone Agg canvas: no GUI backend or pyplot registry mutation.
                with patch.object(artifacts.analysis.plt, "figure", side_effect=agg_figure):
                    figure = artifacts.analysis.draw_taxonomy(taxonomy)
                self.addCleanup(figure.clear)
                figure.canvas.draw()
                components = sorted(
                    (c for c in nx.weakly_connected_components(taxonomy) if len(c) > 1),
                    key=lambda c: (-len(c), tuple(sorted(c))),
                )
                for ax, component in zip(figure.axes, components):
                    boxes = {
                        t.get_text().split("\n")[0]: t.get_bbox_patch().get_window_extent()
                        for t in ax.texts if t.get_text().split("\n")[0] in component
                    }
                    edges = [(u,v) for u in sorted(component)
                             for v in sorted(taxonomy.successors(u)) if v in component]
                    arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
                    for (u,v), arrow in zip(edges, arrows):
                        for segment, _ in arrow.get_path().iter_bezier():
                            points = ax.transData.transform(segment(np.linspace(0,1,80)))
                            for n, box in boxes.items():
                                if n not in (u,v):
                                    self.assertFalse(
                                        any(box.contains(x,y) for x,y in points), (stage,u,v,n)
                                    )

    def test_cli_integrates_all_stages_and_detects_stale_artifacts_without_writes(self):
        with tempfile.TemporaryDirectory(prefix="ha2-artifacts-") as tmp:
            data = Path(tmp)
            (data / "raw").mkdir()
            for name in input_sync.RAW_FILES.values():
                shutil.copyfile(artifacts.ROOT / "raw" / name, data / "raw" / name)
            for name in ["pos_annotations.csv", "requirements.txt", "HA2_Semantic_Network.ipynb",
                         *artifacts.DOC_FILES]:
                shutil.copyfile(artifacts.ROOT / name, data / name)
            (data / "archive").mkdir()
            archived = data / "archive/preserve.txt"
            archived.write_text("Unrelated historical artifact")
            raw_before = {p.name: p.read_bytes() for p in (data / "raw").iterdir()}
            input_sync.prepare(data)
            inputs_before = {name: (data / name).read_bytes() for name in ("nodes.csv","edges.csv")}
            command = [sys.executable, str(artifacts.ROOT / "ha2_artifacts.py"), "--data-dir", tmp]
            run = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(run.stderr, "")
            result = json.loads(run.stdout)
            self.assertEqual(result["figure_files"], 16)
            self.assertFalse(result["interpretation_written"])
            self.assertEqual(archived.read_text(), "Unrelated historical artifact")
            self.assertEqual(raw_before, {p.name: p.read_bytes() for p in (data / "raw").iterdir()})
            self.assertEqual(inputs_before, {name: (data / name).read_bytes() for name in inputs_before})
            self.assertEqual(verify_export(data / "results/perturbation")["verified_deltas"], 264)
            manifest_path = data / "results/artifact_manifest.json"
            before = manifest_path.read_bytes()
            manifest = json.loads(before)
            self.assertFalse(any("archive/" in p for p in manifest["current_output_sha256"]))
            self.assertEqual(manifest["notebook_source_sha256"],
                             artifacts.notebook_source_sha256(data / "HA2_Semantic_Network.ipynb"))
            for stem in artifacts.FIGURE_STEMS:
                png, svg = (data / f"figures/{stem}.{suffix}" for suffix in ("png","svg"))
                self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertIn("<svg", svg.read_text())
                self.assertTrue(all(line == line.rstrip() for line in svg.read_text().splitlines()))
            verified = subprocess.run(command + ["--check"], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(verified.stdout), result)
            artifacts.write_artifact_manifest(data)
            self.assertEqual(before, manifest_path.read_bytes())
            # A corrupted figure must be rejected, not silently regenerated by --check.
            changed = data / "figures/perturbation_deltas.png"
            changed.write_bytes(changed.read_bytes() + b"changed")
            damaged = changed.read_bytes()
            failed = subprocess.run(command + ["--check"], capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertIn("Artifact fingerprint mismatch", failed.stderr)
            self.assertEqual(changed.read_bytes(), damaged)
            self.assertEqual(before, manifest_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
