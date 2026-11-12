from __future__ import annotations

from collections.abc import Callable, Iterable

import duckdb
import numpy as np
import pyarrow as pa

from ced.model.schemas import IsolationTreePruningReport

EdgeBatches = Callable[[], Iterable[tuple[np.ndarray, np.ndarray]]]
BATCH_ROWS = 1_000_000


def _compress(labels: np.ndarray) -> np.ndarray:
    """Follow parent pointers until every component points directly to its minimum."""
    while True:
        parents = labels[labels]
        if np.array_equal(parents, labels):
            return labels
        labels = parents


def _isolated_tree_nodes(node_count: int, max_nodes: int, batches: EdgeBatches) -> tuple[np.ndarray, int, int]:
    """Return the node mask and component/statement counts for small disconnected trees.

    Labels converge by undirected minimum-label propagation with pointer jumping. Each
    pass processes Arrow-sized arrays rather than Python statement rows, and storage is
    linear in the node count. A connected multigraph is a tree exactly when it has one
    fewer edge than nodes, so parallel statements and self-loops prevent pruning.
    ``batches`` must return a fresh iterable for each convergence and counting pass.
    """
    if node_count < 0 or max_nodes < 1:
        raise ValueError("node_count must be nonnegative and max_nodes must be positive")
    labels = np.arange(node_count, dtype=np.int64)
    while True:
        previous = labels.copy()
        for head, tail in batches():
            if head.shape != tail.shape:
                raise ValueError("endpoint batches must have matching shapes")
            if head.size == 0:
                continue
            if head.min() < 0 or tail.min() < 0 or head.max() >= node_count or tail.max() >= node_count:
                raise ValueError("statement endpoint is outside the node vocabulary")
            head_roots = labels[head]
            tail_roots = labels[tail]
            np.minimum.at(labels, head, tail_roots)
            np.minimum.at(labels, tail, head_roots)
        labels = _compress(labels)
        if np.array_equal(labels, previous):
            break

    node_counts = np.bincount(labels, minlength=node_count)
    edge_counts = np.zeros(node_count, dtype=np.int64)
    for head, _tail in batches():
        np.add.at(edge_counts, labels[head], 1)
    pruned_roots = (node_counts <= max_nodes) & (edge_counts == node_counts - 1)
    return (
        pruned_roots[labels],
        int(np.count_nonzero(pruned_roots)),
        int(edge_counts[pruned_roots].sum()),
    )


def _count(connection: duckdb.DuckDBPyConnection, table: str) -> int:
    row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def prune_isolated_trees(connection: duckdb.DuckDBPyConnection, max_nodes: int) -> IsolationTreePruningReport:
    """Create ``pruned_nodes`` and clean up all temporary indexes on every path.

    Connectivity ignores statement direction and includes every relation role before
    any train/validation/test assignment. The caller filters both endpoints and all
    downstream vocabularies against ``pruned_nodes``.
    """
    input_nodes = _count(connection, "nodes")
    input_statements = _count(connection, "statements")
    if max_nodes == 0:
        connection.execute("CREATE TABLE pruned_nodes AS SELECT id FROM nodes WHERE false")
        return IsolationTreePruningReport(
            max_component_nodes=0,
            input_nodes=input_nodes,
            input_statements=input_statements,
            pruned_components=0,
            pruned_nodes=0,
            pruned_statements=0,
        )

    connection.execute(
        """CREATE TABLE indexed_nodes AS
        SELECT id, CAST(row_number() OVER (ORDER BY id) - 1 AS BIGINT) AS idx FROM nodes"""
    )
    try:
        connection.execute(
            """CREATE TABLE endpoint_index AS
            SELECT h.idx AS head, t.idx AS tail
            FROM statements s
            JOIN indexed_nodes h ON s.subject = h.id
            JOIN indexed_nodes t ON s.object = t.id"""
        )
        try:
            if _count(connection, "endpoint_index") != input_statements:
                raise ValueError("statement endpoint count mismatch during isolated-tree pruning")

            def batches() -> Iterable[tuple[np.ndarray, np.ndarray]]:
                reader = connection.execute("SELECT head, tail FROM endpoint_index").to_arrow_reader(BATCH_ROWS)
                for batch in reader:
                    yield (
                        batch.column(0).to_numpy(zero_copy_only=False),
                        batch.column(1).to_numpy(zero_copy_only=False),
                    )

            node_mask, component_count, statement_count = _isolated_tree_nodes(input_nodes, max_nodes, batches)
            indices = pa.table({"idx": pa.array(np.flatnonzero(node_mask), type=pa.int64())})
            connection.register("pruned_node_indices", indices)
            try:
                connection.execute(
                    """CREATE TABLE pruned_nodes AS
                    SELECT n.id FROM indexed_nodes n JOIN pruned_node_indices p ON n.idx = p.idx"""
                )
            finally:
                connection.unregister("pruned_node_indices")
        finally:
            connection.execute("DROP TABLE endpoint_index")
    finally:
        connection.execute("DROP TABLE indexed_nodes")
    return IsolationTreePruningReport(
        max_component_nodes=max_nodes,
        input_nodes=input_nodes,
        input_statements=input_statements,
        pruned_components=component_count,
        pruned_nodes=int(np.count_nonzero(node_mask)),
        pruned_statements=statement_count,
    )
