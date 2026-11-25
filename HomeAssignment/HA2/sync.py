"""Build analysis CSVs from the current four local HA2 exports."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def read(name):
    with (RAW / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    with (ROOT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    concepts, nodes, edges, sources = (
        read(name) for name in ("Concepts_hw2.csv", "Nodes.csv", "Edges.csv", "Sources.csv")
    )
    targets = {r["node_id"]: r["label"] for r in nodes if r["is_target"] == "TRUE"}
    if len(targets) != 25 or len(concepts) != 25 or len(sources) != 25:
        raise ValueError("Expected 25 target concepts and source records")
    for rows in (concepts, sources):
        if {r["concept_id"]: r["term"] for r in rows} != targets:
            raise ValueError("Target IDs and terms differ")
    # Match the strictly spatial definition and example supplied for близость.
    proximity = next(r for r in concepts if r["term"] == "близость")
    proximity["social"] = "0"
    proximity["sceptic"] = "Social=0 for the spatial sense. The source proposal assigns 1 only in interpersonal uses."
    orientation = next(r for r in concepts if r["term"] == "ориентация")
    orientation["definition"] = "Определение своего положения и направлений относительно пространственных ориентиров."
    orientation["sceptic"] = "Definition aligned with the process sense (ILI i109572) and the supplied example."
    write("concepts.csv", concepts)
    write("nodes.csv", nodes)
    write("edges.csv", edges)
    # These labels are lexical annotations. ILI identifiers do not encode POS.
    write("pos_annotations.csv", [
        dict(node_id=r["node_id"], POS="NOUN", provenance="manual",
             justification="Noun label in the current Russian inventory.")
        for r in nodes if not r["source_id"].endswith("-N")
    ])


if __name__ == "__main__":
    main()
