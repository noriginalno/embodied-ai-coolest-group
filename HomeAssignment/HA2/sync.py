"""Refresh role 1 exports and rebuild the role 2 edge list."""
import argparse
import csv
import io
from collections import Counter
from graphlib import TopologicalSorter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
HA1_EDGES = ROOT.parent / "HA1_submission/data/ruwordnet_edges.csv"

HA1_LABELS = {
    "106509-N": "СОСТАВНАЯ ЧАСТЬ", "106581-N": "ПОЛОСА НА ПОВЕРХНОСТИ, В ПРОСТРАНСТВЕ",
    "149355-N": "ПОЛОСА НА ПОВЕРХНОСТИ", "106768-N": "СВОЙСТВО, ХАРАКТЕРИСТИКА",
    "146671-N": "АБСТРАКТНАЯ СУЩНОСТЬ", "147134-N": "ФИЗИЧЕСКАЯ СУЩНОСТЬ",
    "153782-N": "ПОСТОЯННАЯ СУЩНОСТЬ",
}

ALIASES = {"A24": ("C01", "120377-N"), "A01": ("C18", "106451-N"),
           "A28": ("C18", "106451-N")}
EXCLUDED = {
    ("C02", "C21"): "Наружная сторона тела и геометрическая плоскость имеют разные смыслы.",
    ("A18", "C24"): "Гипероним ДЛИНА относится к величине. В графе радиус представлен отрезком.",
}
for node_id in ("A12", "A13", "A14", "A15", "A16"):
    EXCLUDED[node_id, "C25"] = "В графе диаметр определён как отрезок."
SHEETS = {
    "Concepts_hw2": {"concept_id", "term"},
    "nodes": {"node_id", "label", "is_target"},
    "Edges": {"parent", "child", "relation", "provenance"},
    "Sources": {"concept_id", "source_id"},
}
URL = "https://docs.google.com/spreadsheets/d/1H9Oi7JLhYowXhTEOvbHaJyApQGAFSvIqZ0h9pW7_QOI/gviz/tq?"
FIELDS = ["source", "relation", "target", "provenance", "confidence", "justification"]
MANUAL = [
    ("координата", "specifies_location_of", "точка",
     "Координата задаёт положение точки по оси выбранной системы координат."),
    ("путь", "has_endpoint", "место",
     "Маршрут связывает исходное и конечное места."),
    ("диаметр", "passes_through", "центр",
     "Диаметр круга или шара проходит через его центр."),
]


def parse(name, data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    if not SHEETS[name].issubset(reader.fieldnames or ()):
        raise ValueError(f"Unexpected columns in {name}")
    return list(reader)


def build(tables):
    nodes = {row["node_id"]: row for row in tables["nodes"]}
    if len(nodes) != len(tables["nodes"]):
        raise ValueError("Duplicate node IDs")
    targets = {key: row["label"] for key, row in nodes.items()
               if row["is_target"].upper() == "TRUE"}
    concepts = {row["concept_id"]: row["term"] for row in tables["Concepts_hw2"]}
    if len(targets) != 25 or len(tables["Concepts_hw2"]) != 25 or targets != concepts:
        raise ValueError("Target nodes and Concepts_hw2 differ")
    if Counter(row["concept_id"] for row in tables["Sources"]) != Counter(targets.keys()):
        raise ValueError("Sources must cover each target exactly once")

    aliases = {}
    for old, (new, synset) in ALIASES.items():
        if old not in nodes:
            continue
        assert nodes[old]["is_target"].upper() == "FALSE", old
        assert nodes[old]["source_id"] == synset, old
        assert synset in nodes[new]["source_id"].split("; "), new
        aliases[old] = new

    expected_labels = {
        "C02": "поверхность", "C21": "плоскость", "C24": "радиус", "C25": "диаметр",
        "A18": "ДЛИНА", "A12": "ВЕЛИЧИНА", "A13": "ЛИНЕЙНЫЙ РАЗМЕР",
        "A14": "ПРОТЯЖЕНИЕ", "A15": "ПРОТЯЖЕННОСТЬ", "A16": "РАЗМЕР",
    }
    output = []
    excluded = []
    for row in tables["Edges"]:
        parent, child = row["parent"], row["child"]
        if parent not in nodes or child not in nodes or row["relation"] != "is_a":
            raise ValueError(f"Invalid taxonomy edge: {row}")
        if (parent, child) in EXCLUDED:
            for key in (parent, child):
                if nodes[key]["label"] != expected_labels[key]:
                    raise ValueError(f"Review exclusion for changed node: {key}")
            excluded.append((parent, child, EXCLUDED[parent, child]))
            continue
        parent, child = aliases.get(parent, parent), aliases.get(child, child)
        justification = (f"Лист Edges роли 1 указывает «{nodes[parent]['label']}» "
                         f"как гипероним для «{nodes[child]['label']}».")
        output.append((parent, "is_a", child, row["provenance"], "medium", justification))

    for key in aliases:
        del nodes[key]

    # Восстанавливаем только пути, уже зафиксированные в HA1.
    with HA1_EDGES.open(encoding="utf-8", newline="") as handle:
        ha1 = [(row["parent_id"], row["child_id"]) for row in csv.DictReader(handle)]
    synset_nodes = {}
    for key, row in nodes.items():
        for source_id in row["source_id"].split("; "):
            if source_id.endswith(("-N", "-A", "-V")):
                if source_id in synset_nodes:
                    raise ValueError(f"Duplicate synset after alias resolution: {source_id}")
                synset_nodes[source_id] = key

    if nodes.get("A04", {}).get("label") == HA1_LABELS["147134-N"] and not nodes["A04"]["source_id"]:
        synset_nodes["147134-N"] = "A04"
        nodes["A04"]["source_id"] = "147134-N"
        nodes["A04"]["source"] = "HA1; RuWordNet"

    def node_for(synset):
        if synset not in synset_nodes:
            key = f"HA1:{synset}"
            synset_nodes[synset] = key
            nodes[key] = dict(node_id=key, label=HA1_LABELS[synset], node_type="auxiliary",
                              is_target="FALSE", source="HA1; RuWordNet", source_id=synset)
        return synset_nodes[synset]

    existing = {row[:3] for row in output}
    pending = list(synset_nodes)
    visited = set()
    while pending:
        child_synset = pending.pop()
        if child_synset in visited:
            continue
        visited.add(child_synset)
        for parent_synset, candidate in ha1:
            if candidate != child_synset:
                continue
            edge = (node_for(parent_synset), "is_a", node_for(child_synset))
            if edge not in existing:
                output.append((*edge, "wordnet", "medium",
                               f"Путь HA1 RuWordNet: {parent_synset} -> {child_synset}."))
                existing.add(edge)
            pending.append(parent_synset)

    graph = TopologicalSorter({key: [] for key in nodes})
    for parent, _, child, *_ in output:
        graph.add(child, parent)
    list(graph.static_order())

    by_term = {term: key for key, term in targets.items()}
    for source, relation, target, justification in MANUAL:
        if source not in by_term or target not in by_term:
            raise ValueError(f"Review manual edge after inventory change: {source} -> {target}")
        output.append((by_term[source], relation, by_term[target], "manual", "high", justification))
    if len({row[:3] for row in output}) != len(output):
        raise ValueError("Duplicate edges")
    return list(nodes.values()), output, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Download fresh role 1 CSVs")
    args = parser.parse_args()
    data = {}
    for sheet in SHEETS:
        if args.refresh:
            url = URL + urlencode({"tqx": "out:csv", "sheet": sheet})
            with urlopen(url, timeout=15) as response:
                data[sheet] = response.read()
        else:
            data[sheet] = (RAW / f"{sheet}.csv").read_bytes()
    tables = {name: parse(name, content) for name, content in data.items()}
    nodes, edges, excluded = build(tables)
    if args.refresh:
        RAW.mkdir(exist_ok=True)
        for name, content in data.items():
            (RAW / f"{name}.csv").write_bytes(content)
    write_csv("nodes.csv", list(nodes[0]), (row.values() for row in nodes))
    write_csv("edges.csv", FIELDS, edges)
    write_csv("excluded_edges.csv", ["parent", "child", "reason"], excluded)
    print(f"{len(nodes)} nodes. Edges: {dict(Counter(row[1] for row in edges))}")


def write_csv(name, fields, rows):
    with (ROOT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)



if __name__ == "__main__":
    main()
