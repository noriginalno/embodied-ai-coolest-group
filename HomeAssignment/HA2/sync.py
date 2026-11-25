"""Prepare the revised local HA2 CSV snapshot; no network access or S calculation."""
import argparse
from collections import Counter
import csv
from graphlib import TopologicalSorter
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
RAW_FILES = {
    "nodes": "Nodes.csv", "Edges": "Edges.csv",
    "Concepts_hw2": "Concepts_hw2.csv", "Sources": "Sources.csv",
}
NODE_FIELDS = ["node_id", "label", "node_type", "is_target", "source", "source_id"]
POS_FIELDS = ["POS", "POS_provenance", "POS_justification"]
EDGE_FIELDS = ["source", "relation", "target", "provenance", "confidence", "justification"]
ANNOTATION_FIELDS = ["node_id", "label", "source_id", "POS", "provenance", "justification"]
SCHEMAS = {
    "nodes": set(NODE_FIELDS), "Edges": set(EDGE_FIELDS),
    "Concepts_hw2": {"concept_id", "term", "definition", "provenance"},
    "Sources": {"concept_id", "term", "source_id"},
}
ALLOWED_PROVENANCE = {"seed", "manual", "wordnet", "framenet"}
PROVENANCE_MAP = {
    **{p: p for p in ALLOWED_PROVENANCE},
    "ruwordnet": "wordnet",
    "ruwordnet extraction": "wordnet",
    "ruwordnet/ha1": "wordnet",
    "ruwordnet page": "wordnet",
    # A gloss/association can support a manually named predicate. It is not
    # evidence that that exact predicate was exported as a WordNet relation.
    "ruwordnet gloss/manual": "manual",
    "ruwordnet association/manual": "manual",
}
ALLOWED_POS = {
    "NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "ADP", "PRON", "NUM", "DET",
    "CCONJ", "SCONJ", "PART", "INTJ", "PUNCT", "SYM", "X", "UNKNOWN",
}
OUTPUT_FIELDS = {
    "nodes.csv": NODE_FIELDS + POS_FIELDS,
    "edges.csv": EDGE_FIELDS + ["source_provenance"],
    "excluded_edges.csv": ["parent", "child", "reason"],
}


def parse_csv(data, required, name):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not set(required) <= set(fields):
        raise ValueError(f"Unexpected columns in {name}: required {sorted(required)}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row in {name}")
    return rows


def parse(name, data):
    return parse_csv(data, SCHEMAS[name], name)


def unique_rows(rows, key, name):
    result = {}
    for row in rows:
        value = row[key]
        if not value or value in result:
            raise ValueError(f"Empty or duplicate {key} in {name}: {value!r}")
        result[value] = row
    return result


def canonical_provenance(value):
    try:
        return PROVENANCE_MAP[value.strip().casefold()]
    except KeyError:
        raise ValueError(f"Unsupported edge provenance: {value!r}") from None


def annotate_pos(nodes, annotations):
    """Bind manual POS to ID + label + source_id, not a recycled ID alone."""
    node_map = unique_rows(nodes, "node_id", "nodes")
    by_id = unique_rows(annotations, "node_id", "pos_annotations.csv")
    extra = set(by_id) - set(node_map)
    if extra:
        raise ValueError(f"POS annotations for absent nodes: {sorted(extra)}")
    result = []
    for node in nodes:
        row = {key: node[key] for key in NODE_FIELDS}
        annotation = by_id.get(row["node_id"])
        suffixes = set(re.findall(r"\d+-(N|V|A)(?=\s*(?:;|$))", row["source_id"]))
        if len(suffixes) > 1:
            raise ValueError(f"Conflicting POS suffixes for {row['node_id']}")
        sourced_pos = {"N": "NOUN", "V": "VERB", "A": "ADJ"}.get(
            next(iter(suffixes), "")
        )
        if annotation:
            if any(annotation[k] != row[k] for k in ("label", "source_id")):
                raise ValueError(f"Stale POS annotation for {row['node_id']}: identity changed")
            if (annotation["POS"] not in ALLOWED_POS
                    or annotation["provenance"] != "manual"
                    or not annotation["justification"].strip()):
                raise ValueError(f"Invalid manual POS annotation for {row['node_id']}")
            if sourced_pos and annotation["POS"] != sourced_pos:
                raise ValueError(f"Manual POS conflicts with synset suffix for {row['node_id']}")
        if sourced_pos:
            row.update(
                POS=sourced_pos, POS_provenance="wordnet:source_id_suffix",
                POS_justification="POS прочитан из суффикса синсета в source_id.",
            )
        elif annotation:
            row.update(
                POS=annotation["POS"], POS_provenance="manual:pos_annotations.csv",
                POS_justification=annotation["justification"],
            )
        else:
            row.update(
                POS="UNKNOWN", POS_provenance="not_supplied",
                POS_justification="POS не передан; по ILI или названию автоматически не выводится.",
            )
        result.append(row)
    return result


def build(tables, annotations=()):
    """Preserve the accepted inventory and EVERY supplied typed edge.

    No historical aliases, exclusions, added parents, or hardcoded manual edges.
    Structural checks protect data transfer; concepts are not semantically reviewed.
    """
    node_map = unique_rows(tables["nodes"], "node_id", "Nodes.csv")
    if not node_map:
        raise ValueError("Nodes.csv is empty")
    for row in node_map.values():
        if not row["label"] or row["node_type"] not in {"target", "auxiliary"}:
            raise ValueError(f"Invalid node fields: {row['node_id']}")
        if row["is_target"] not in {"TRUE", "FALSE"}:
            raise ValueError(f"Invalid is_target: {row['node_id']}")
        if (row["node_type"] == "target") != (row["is_target"] == "TRUE"):
            raise ValueError(f"node_type and is_target disagree: {row['node_id']}")
    concepts = unique_rows(tables["Concepts_hw2"], "concept_id", "Concepts_hw2.csv")
    sources = unique_rows(tables["Sources"], "concept_id", "Sources.csv")
    targets = {n: r for n, r in node_map.items() if r["is_target"] == "TRUE"}
    if {n: r["label"] for n, r in targets.items()} != {
        n: r["term"] for n, r in concepts.items()
    }:
        raise ValueError("Target IDs/labels differ between Nodes.csv and Concepts_hw2.csv")
    if set(sources) != set(targets) or any(
        sources[n]["term"] != row["label"] or sources[n]["source_id"] != row["source_id"]
        for n, row in targets.items()
    ):
        raise ValueError("Sources.csv identities differ from target nodes")
    edges, seen = [], set()
    taxonomy = {n: [] for n in node_map}
    for raw in tables["Edges"]:
        if any(not raw[field].strip() for field in EDGE_FIELDS):
            raise ValueError("Empty required edge field")
        source, target = raw["source"], raw["target"]
        if source not in node_map or target not in node_map:
            raise ValueError(f"Unknown edge endpoint: {source} -> {target}")
        key = source, raw["relation"], target
        if key in seen:
            raise ValueError(f"Duplicate typed edge: {key}")
        seen.add(key)
        row = {field: raw[field] for field in EDGE_FIELDS}
        row["provenance"] = canonical_provenance(raw["provenance"])
        row["source_provenance"] = raw["provenance"]
        edges.append(row)
        if row["relation"] == "is_a":
            taxonomy[target].append(source)  # TopologicalSorter expects predecessors.
    list(TopologicalSorter(taxonomy).static_order())
    return annotate_pos(list(node_map.values()), annotations), edges, []


def build_local(data_dir=ROOT):
    data_dir = Path(data_dir)
    tables = {
        name: parse(name, (data_dir / "raw" / filename).read_bytes())
        for name, filename in RAW_FILES.items()
    }
    annotations = parse_csv(
        (data_dir / "pos_annotations.csv").read_bytes(),
        ANNOTATION_FIELDS, "pos_annotations.csv",
    )
    return build(tables, annotations)


def load_prepared(data_dir=ROOT):
    """Reject stale prepared tables before old IDs can acquire new definitions."""
    data_dir = Path(data_dir)
    expected_nodes, expected_edges, _ = build_local(data_dir)
    prepared = []
    for name, expected, keys in (
        ("nodes.csv", expected_nodes, ["node_id"]),
        ("edges.csv", expected_edges, ["source", "relation", "target"]),
    ):
        try:
            actual = parse_csv((data_dir / name).read_bytes(), OUTPUT_FIELDS[name], name)
        except ValueError as exc:
            raise ValueError(
                f"{name} is stale; run python3 HomeAssignment/HA2/sync.py"
            ) from exc
        index = lambda rows: {tuple(row[k] for k in keys): row for row in rows}
        if len(actual) != len(expected) or index(actual) != index(expected):
            raise ValueError(f"{name} is stale or edited; run python3 HomeAssignment/HA2/sync.py")
        prepared.append(actual)
    return tuple(prepared)


def csv_bytes(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def prepare(data_dir=ROOT):
    data_dir = Path(data_dir)
    nodes, edges, excluded = build_local(data_dir)  # Validate everything before writing.
    outputs = {
        name: csv_bytes(OUTPUT_FIELDS[name], rows)
        for name, rows in (("nodes.csv", nodes), ("edges.csv", edges),
                           ("excluded_edges.csv", excluded))
    }
    paths = [data_dir / "raw" / filename for filename in RAW_FILES.values()]
    paths.append(data_dir / "pos_annotations.csv")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=data_dir, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    manifest = {
        "input_format_version": 2, "source_snapshot_commit": commit,
        "node_count": len(nodes),
        "target_count": sum(n["is_target"] == "TRUE" for n in nodes),
        "edge_count": len(edges),
        "relation_counts": dict(Counter(e["relation"] for e in edges)),
        "provenance_counts": dict(Counter(e["provenance"] for e in edges)),
        "POS_counts": dict(Counter(n["POS"] for n in nodes)),
        "POS_provenance_counts": dict(Counter(n["POS_provenance"] for n in nodes)),
        "input_sha256": {
            str(p.relative_to(data_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "prepared_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in outputs.items()
        },
        "import_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "downstream_results_status": "see results/run_manifest.json; importer does not calculate S",
    }
    for name, content in outputs.items():
        (data_dir / name).write_bytes(content)
    (data_dir / "prepared_inputs.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="Check prepared inputs without writing")
    parser.add_argument("--refresh", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.refresh:
        parser.error("--refresh is disabled: revised local CSVs are authoritative; no old Sheets download")
    if args.check:
        nodes, edges = load_prepared(args.data_dir)
        print(f"Inputs current: {len(nodes)} nodes, {len(edges)} edges.")
    else:
        manifest = prepare(args.data_dir)
        print(json.dumps({
            k: manifest[k] for k in (
                "node_count", "target_count", "edge_count", "relation_counts",
                "provenance_counts", "POS_counts", "downstream_results_status",
            )
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
