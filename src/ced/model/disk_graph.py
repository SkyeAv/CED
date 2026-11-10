from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ced.model.batches import SampledNeighborhood


class DiskGraph:
    """Indexed positives, train adjacency, and typed pools with bounded Python memory."""

    def __init__(self, path: Path, arrays: Mapping[str, np.ndarray]) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.relation_count = int(arrays["relation_id"].max()) + 1 if arrays["relation_id"].size else 0
        self.connection.executescript("""
          CREATE TABLE positives(h INTEGER, r INTEGER, t INTEGER, PRIMARY KEY(h,r,t)) WITHOUT ROWID;
          CREATE TABLE adjacency(node INTEGER, statement INTEGER, other INTEGER, r INTEGER);
          CREATE TABLE entities(id INTEGER PRIMARY KEY, type INTEGER);
          CREATE TABLE domains(r INTEGER, side INTEGER, type INTEGER, PRIMARY KEY(r,side,type)) WITHOUT ROWID;
        """)
        types = arrays["type_set_id"]
        for start in range(0, len(types), 5000):
            self.connection.executemany(
                "INSERT INTO entities VALUES (?,?)",
                ((i, int(types[i])) for i in range(start, min(start + 5000, len(types)))),
            )
        edges, relations, splits = (arrays[key] for key in ("edge_index", "relation_id", "split"))
        for start in range(0, len(relations), 5000):
            end = min(start + 5000, len(relations))
            self.connection.executemany(
                "INSERT OR IGNORE INTO positives VALUES (?,?,?)",
                ((int(edges[0, i]), int(relations[i]), int(edges[1, i])) for i in range(start, end)),
            )
            for i in range(start, end):
                if splits[i] != 0:
                    continue
                h, r, t = int(edges[0, i]), int(relations[i]), int(edges[1, i])
                self.connection.executemany("INSERT INTO adjacency VALUES (?,?,?,?)", ((h, i, t, r), (t, i, h, r)))
                self.connection.executemany(
                    "INSERT OR IGNORE INTO domains VALUES (?,?,?)", ((r, 0, int(types[h])), (r, 1, int(types[t])))
                )
        self.connection.executescript("""
          CREATE INDEX adjacency_node ON adjacency(node,statement);
          CREATE INDEX entity_type ON entities(type,id);
          CREATE TABLE pools AS SELECT d.r,d.side,e.id FROM domains d JOIN entities e ON d.type=e.type;
          CREATE INDEX pool_key ON pools(r,side,id);
          CREATE TABLE pool_counts AS SELECT r,side,count(*) AS n FROM pools GROUP BY r,side;
        """)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def negatives(
        self, triples: list[tuple[int, int, int]], count: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        heads = np.zeros((len(triples), count), dtype=np.int64)
        tails = np.zeros_like(heads)
        valid = np.zeros_like(heads, dtype=np.bool_)
        random = np.random.default_rng(seed)
        for row, (h, r, t) in enumerate(triples):
            heads[row, :], tails[row, :] = h, t
            for column in range(count):
                side = column % 2
                size = self.connection.execute("SELECT n FROM pool_counts WHERE r=? AND side=?", (r, side)).fetchone()
                if not size or not size[0]:
                    continue
                offset = int(random.integers(size[0]))
                candidate = self.connection.execute(
                    "SELECT id FROM pools WHERE r=? AND side=? ORDER BY id LIMIT 1 OFFSET ?", (r, side, offset)
                ).fetchone()
                assert candidate is not None
                pivot = candidate[0]
                # Two indexed ranges wrap around; NOT EXISTS filters every known positive.
                for comparison, boundary in ((">=", pivot), ("<", pivot)):
                    query = (
                        "SELECT p.id FROM pools p WHERE p.r=? AND p.side=? AND p.id "
                        + comparison
                        + " ? AND NOT EXISTS (SELECT 1 FROM positives k WHERE k.r=? AND k.h="
                        + ("p.id" if side == 0 else "?")
                        + " AND k.t="
                        + ("?" if side == 0 else "p.id")
                        + ") ORDER BY p.id LIMIT 1"
                    )
                    found = self.connection.execute(query, (r, side, boundary, r, t if side == 0 else h)).fetchone()
                    if found:
                        heads[row, column], tails[row, column] = (found[0], t) if side == 0 else (h, found[0])
                        valid[row, column] = True
                        break
        return heads, tails, valid

    def neighborhood(
        self,
        edges: np.ndarray,
        relations: np.ndarray,
        seeds: np.ndarray,
        fanout: list[int],
        seed: int,
        node_limit: int,
        edge_types: np.ndarray | None = None,
    ) -> SampledNeighborhood:
        nodes = set(map(int, seeds))
        if len(nodes) > node_limit:
            raise ValueError("seed and negative endpoints exceed batch node limit")
        frontier = sorted(nodes)
        selected = set()
        random = np.random.default_rng(seed)
        for limit in fanout:
            if limit < 0:
                raise ValueError("neighbor fanout must be nonnegative")
            following = set()
            for node in frontier:
                # Reservoir sampling bounds memory even for high-degree nodes.
                reservoir = []
                for position, (statement, other) in enumerate(
                    self.connection.execute(
                        "SELECT statement,other FROM adjacency WHERE node=? ORDER BY statement", (node,)
                    )
                ):
                    if position < limit:
                        reservoir.append((statement, other))
                    elif limit:
                        replace = int(random.integers(position + 1))
                        if replace < limit:
                            reservoir[replace] = (statement, other)
                for statement, other in sorted(reservoir):
                    if other not in nodes and len(nodes) >= node_limit:
                        continue
                    selected.add(statement)
                    if other not in nodes:
                        following.add(other)
                        nodes.add(other)
            frontier = sorted(following)
        global_ids = np.array(sorted(nodes), dtype=np.int64)
        statement_ids = np.array(sorted(selected), dtype=np.int64)
        local = np.searchsorted(global_ids, edges[:, statement_ids]).astype(np.int64)
        parameter_ids = np.zeros(len(relations), dtype=np.int64) if edge_types is None else edge_types
        return SampledNeighborhood(
            global_ids,
            np.concatenate((local, local[::-1]), axis=1),
            np.tile(statement_ids, 2),
            np.concatenate((relations[statement_ids], relations[statement_ids] + self.relation_count)),
            np.tile(parameter_ids[statement_ids], 2),
        )
