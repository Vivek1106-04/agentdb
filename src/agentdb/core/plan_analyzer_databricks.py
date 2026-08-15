"""Normalizing a Databricks plan into the IR (SPEC §7, §8.2).

``EXPLAIN FORMATTED`` returns text, not JSON, in two parts: a tree of numbered
nodes, then one detail block per node. Both are parsed here, because the pruning
evidence is split across them — the tree says what the shape is, the detail
blocks say which predicates the scan could push to per-file statistics and how
many files it ended up reading.

Two properties matter more than completeness:

* **Photon is inferred, never claimed.** Databricks reports a fallback only as
  the *absence* of a ``Photon`` prefix on a node name. This module marks each
  node accordingly and says so; it never pretends the engine reported a flag.
* **``files_total`` does not come from the plan.** The plan says how many files
  were read, not how many exist. The total comes from ``DESCRIBE DETAIL`` and is
  passed in, so a pruning ratio is computed from two measured numbers or not at
  all.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace

from agentdb.adapters import RawPlan
from agentdb.core.plan_ir import PlanNode, PlanOp, PlanSummary

PHYSICAL_PLAN_HEADER = "== Physical Plan =="

_TREE_LINE = re.compile(r"^(?P<indent>[\s+\-:*|]*)(?P<label>[A-Za-z].*?)\s*\((?P<id>\d+)\)\s*$")
_DETAIL_HEADER = re.compile(r"^\((?P<id>\d+)\)\s+(?P<label>.+?)\s*$")
_DETAIL_FIELD = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 _/]*?)\s*:\s*(?P<value>.*)$")
_ATTRIBUTE_ID = re.compile(r"#\d+L?")
_STATISTICS = re.compile(
    r"(?P<bytes>[\d.]+)\s*(?P<unit>[KMGTP]?i?B)(?:,\s*(?P<rows>[\d.]+)\s*rows)?"
)

_BYTE_UNITS: Mapping[str, int] = {
    "B": 1,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
    "PiB": 1024**5,
}

JOIN_STRATEGIES: Mapping[str, str] = {
    "broadcasthashjoin": "broadcast_hash",
    "broadcastnestedloopjoin": "broadcast_nested_loop",
    "sortmergejoin": "sort_merge",
    "shuffledhashjoin": "shuffle_hash",
    "shufflehashjoin": "shuffle_hash",
}
"""Plan node class to the IR's join vocabulary. The class name is the only place
Databricks states which side it decided to build."""

PHOTON_PREFIX = "Photon"

WRAPPER_NODES = frozenset({"AdaptiveSparkPlan"})
"""Plan wrappers, not operators.

``AdaptiveSparkPlan`` is the root every Databricks plan is wrapped in and it
never carries a ``Photon`` prefix. Counting it as a fallback would report 93%
Photon coverage on a plan that is entirely Photon — observed on the first live
run."""

PUSHED_FILTER_KEYS = ("PushedFilters", "RequiredDataFilters")
"""Where a scan states the predicates it pushed into the read.

Two spellings for one idea, and the second was learned the hard way: a
non-Photon ``FileScan`` prints ``PushedFilters``, while ``PhotonScan`` prints
``RequiredDataFilters`` and no ``PushedFilters`` line at all. A parser that knew
only the documented spelling read every Photon plan as having pushed nothing."""

DATA_FILTER_KEYS = ("DataFilters", "DictionaryFilters")
"""Predicates evaluated below the file-skipping layer.

``DictionaryFilters`` is Photon's parquet dictionary-level filtering: narrower
than a statistics-based file skip, wider than a row scan."""


class PlanParseError(ValueError):
    """The plan output could not be read.

    Raised rather than absorbed, for the same reason as its ClickHouse
    counterpart: a half-understood plan produces warnings nobody can trust.
    """


def parse_plan(raw: RawPlan) -> PlanNode:
    """Turn ``EXPLAIN FORMATTED`` output into a normalized tree."""
    tree_lines, details = _split(raw.payload)
    if not tree_lines:
        raise PlanParseError("plan output has no physical plan tree")
    return _build(tree_lines, details)


def summarize(raw: RawPlan, *, files_total: Mapping[str, int] | None = None) -> PlanSummary:
    """Parse ``raw`` and aggregate its evidence across scans.

    ``files_total`` maps a relation name to the file count ``DESCRIBE DETAIL``
    reported. Without it the plan alone cannot say what fraction was pruned —
    only how many files were read — so the ratio stays ``None`` rather than
    becoming a number with no denominator.
    """
    root = _attach_totals(parse_plan(raw), files_total or {})
    nodes = root.walk()
    scans = tuple(node for node in nodes if node.op is PlanOp.SCAN)

    # Only scans that reported *both* numbers may contribute. Treating an
    # unreported ``files_selected`` as zero is how a plan that measured nothing
    # comes back claiming it pruned everything — observed on the first live run,
    # where a Photon plan carries no file counts at all and the summary read
    # "0.0% of files read after pruning".
    measured = [scan for scan in scans if scan.files_total and scan.files_selected is not None]
    considered = sum(scan.files_total or 0 for scan in measured)
    selected = sum(scan.files_selected or 0 for scan in measured)
    ratio = selected / considered if considered else None

    return PlanSummary(
        root=root,
        engine=raw.engine,
        sql=raw.sql,
        pruning_ratio=ratio,
        pruning_unit="file" if considered else None,
        pruning_source="estimated" if ratio is not None else None,
        full_scan_relations=tuple(
            scan.relation
            for scan in scans
            if scan.relation is not None and scan.pruning_ratio == 1.0
        ),
        estimated_bytes_read=_total_bytes(scans),
        photon_coverage=_photon_coverage(nodes),
    )


def classify(label: str) -> PlanOp:
    """Map a Databricks node class onto the shared operation vocabulary."""
    name = _class_name(label)
    lowered = name.lower()
    if "scan" in lowered and "columnartorow" not in lowered:
        return PlanOp.SCAN
    if "join" in lowered:
        return PlanOp.JOIN
    if "agg" in lowered:
        return PlanOp.AGGREGATE
    if lowered.startswith("sort") or lowered == "photonsort":
        return PlanOp.SORT
    if "exchange" in lowered or "shuffle" in lowered or "broadcastquerystage" in lowered:
        return PlanOp.EXCHANGE
    if "limit" in lowered or "takeordered" in lowered:
        return PlanOp.LIMIT
    if lowered.startswith(("filter", "photonfilter")):
        return PlanOp.FILTER
    return PlanOp.OTHER


def _split(payload: str) -> tuple[list[tuple[int, int, str]], dict[int, dict[str, str]]]:
    """Separate the plan tree from the per-node detail blocks.

    Returns ``(indent, node id, label)`` triples for the tree, and the detail
    fields keyed by node id.
    """
    lines = payload.splitlines()
    start = next(
        (index + 1 for index, line in enumerate(lines) if PHYSICAL_PLAN_HEADER in line),
        0,
    )

    tree: list[tuple[int, int, str]] = []
    details: dict[int, dict[str, str]] = {}
    current: dict[str, str] | None = None
    in_details = False

    for line in lines[start:]:
        header = _DETAIL_HEADER.match(line)
        if header is not None:
            in_details = True
            current = {"__label__": header.group("label")}
            details[int(header.group("id"))] = current
            continue

        if in_details:
            field = _DETAIL_FIELD.match(line)
            if field is not None and current is not None:
                current[field.group("key").strip()] = field.group("value").strip()
            continue

        match = _TREE_LINE.match(line)
        if match is not None:
            tree.append((len(match.group("indent")), int(match.group("id")), match.group("label")))
    return tree, details


def _build(tree: list[tuple[int, int, str]], details: dict[int, dict[str, str]]) -> PlanNode:
    """Rebuild the tree from indentation, deepest lines first.

    The output lists a parent above its children with increasing indentation, so
    a node's children are the following lines indented further than it, up to the
    next line at its own depth or shallower.
    """
    root_indent, root_id, root_label = tree[0]
    children: list[PlanNode] = []
    index = 1
    while index < len(tree):
        indent = tree[index][0]
        if indent <= root_indent:
            break
        end = index + 1
        while end < len(tree) and tree[end][0] > indent:
            end += 1
        children.append(_build(tree[index:end], details))
        index = end
    return _node(root_label, details.get(root_id, {}), tuple(children))


def _node(label: str, fields: Mapping[str, str], children: tuple[PlanNode, ...]) -> PlanNode:
    op = classify(label)
    estimated_bytes, estimated_rows = _statistics(fields.get("Statistics"))
    return PlanNode(
        op=op,
        node_type=label.strip(),
        relation=_relation(label) if op is PlanOp.SCAN else None,
        estimated_rows=estimated_rows,
        estimated_cost=float(estimated_bytes) if estimated_bytes is not None else None,
        filters=_bracket_list(fields.get("Condition") or fields.get("Filter")),
        files_selected=_count(fields.get("number of files read")),
        partition_filters=_bracket_list(fields.get("PartitionFilters")),
        pushed_filters=_first_list(fields, PUSHED_FILTER_KEYS),
        data_filters=_first_list(fields, DATA_FILTER_KEYS),
        photon=label.strip().startswith(PHOTON_PREFIX),
        join_strategy=_join_strategy(label),
        children=children,
    )


def _attach_totals(node: PlanNode, files_total: Mapping[str, int]) -> PlanNode:
    """Rebuild the tree with each scan's file count from ``DESCRIBE DETAIL``.

    The plan reports how many files were *read*; the denominator is a property of
    the table, so it is joined in here rather than guessed from the plan.
    """
    rebuilt = replace(
        node, children=tuple(_attach_totals(child, files_total) for child in node.children)
    )
    if rebuilt.op is not PlanOp.SCAN:
        return rebuilt
    total = _lookup(rebuilt.relation, files_total)
    return rebuilt if total is None else replace(rebuilt, files_total=total)


def _lookup(relation: str | None, files_total: Mapping[str, int]) -> int | None:
    """Match a plan's relation name against the catalogue's, however qualified."""
    if relation is None:
        return None
    if relation in files_total:
        return files_total[relation]
    return files_total.get(relation.rpartition(".")[2])


def _photon_coverage(nodes: tuple[PlanNode, ...]) -> float | None:
    """Fraction of operator nodes that ran on Photon, inferred from node names.

    Wrappers are excluded: ``AdaptiveSparkPlan`` is not an operator and never
    carries the prefix, so counting it would report a fallback on a plan that is
    entirely vectorized.
    """
    operators = [
        node
        for node in nodes
        if node.photon is not None and _class_name(node.node_type) not in WRAPPER_NODES
    ]
    if not operators:
        return None
    return sum(1 for node in operators if node.photon) / len(operators)


def _total_bytes(scans: tuple[PlanNode, ...]) -> int | None:
    sizes = [int(scan.estimated_cost) for scan in scans if scan.estimated_cost is not None]
    return sum(sizes) if sizes else None


def _relation(label: str) -> str | None:
    """The relation a scan reads, as Databricks names it in the plan.

    ``Scan parquet samples.tpch.lineitem`` and ``PhotonScan parquet
    samples.tpch.lineitem`` both name it in the last word.
    """
    parts = label.strip().split()
    if len(parts) < 2:
        return None
    candidate = parts[-1]
    return candidate if "." in candidate or candidate.isidentifier() else None


def _join_strategy(label: str) -> str | None:
    """Which side the planner decided to build, from the node class alone.

    The Photon prefix is stripped first: ``PhotonBroadcastHashJoin`` and
    ``BroadcastHashJoin`` are the same decision made by two executors, and
    reporting only one of them would hide the strategy on exactly the plans this
    project cares most about.
    """
    name = _class_name(label).lower().removeprefix(PHOTON_PREFIX.lower())
    return JOIN_STRATEGIES.get(name)


def _class_name(label: str) -> str:
    return label.strip().split()[0] if label.strip() else ""


def _first_list(fields: Mapping[str, str], keys: Sequence[str]) -> tuple[str, ...]:
    """The first of ``keys`` this node actually printed.

    Databricks spells the same idea differently per executor, and a scan states
    one spelling or the other, never both.
    """
    for key in keys:
        value = fields.get(key)
        if value is not None:
            return _bracket_list(value)
    return ()


def _bracket_list(value: str | None) -> tuple[str, ...]:
    """Split a ``[a, b, c]`` field into its top-level terms.

    Terms are function calls as often as not — ``isnotnull(l_shipdate#2)`` — so
    the split follows bracket depth, and Spark's ``#42`` attribute ids are
    stripped because they are internal identifiers, not column names.
    """
    if not value:
        return ()
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return ()

    terms: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "," and depth == 0:
            terms.append("".join(current).strip())
            current = []
            continue
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        current.append(char)
    terms.append("".join(current).strip())
    return tuple(_ATTRIBUTE_ID.sub("", term) for term in terms if term.strip())


def _count(value: str | None) -> int | None:
    """A plan metric such as ``number of files read: 40``."""
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", value.split()[0]) if value.split() else ""
    return int(digits) if digits else None


def _statistics(value: str | None) -> tuple[int | None, int | None]:
    """``Statistics: 1.2 GiB, 6001215 rows`` into bytes and rows."""
    if value is None:
        return None, None
    match = _STATISTICS.search(value)
    if match is None:
        return None, None
    size = float(match.group("bytes")) * _BYTE_UNITS.get(match.group("unit"), 1)
    rows = match.group("rows")
    return int(size), int(float(rows)) if rows else None
