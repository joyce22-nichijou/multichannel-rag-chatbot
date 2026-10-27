import json
import sqlite3
from pathlib import Path
from typing import Self

import sqlite_vec


class VectorStore:
    """Store chunks and their vectors in a single SQLite file, using sqlite-vec."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self._conn.close()

    def rebuild(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        """Delete the old index, then store every chunk with its vector."""
        dimension = len(vectors[0])
        with self._conn:  # one transaction: all or nothing
            self._conn.execute("BEGIN")  # also covers DROP/CREATE, not only INSERT
            self._conn.execute("DROP TABLE IF EXISTS chunks")
            self._conn.execute("DROP TABLE IF EXISTS vec_chunks")
            self._conn.execute(
                """
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    chunk_id TEXT UNIQUE,
                    text TEXT,
                    headings TEXT,
                    pages TEXT,
                    source TEXT
                )
                """
            )
            self._conn.execute(
                f"CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{dimension}])"
            )
            for row_id, (chunk, vector) in enumerate(
                zip(chunks, vectors, strict=True), 1
            ):
                self._conn.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        row_id,
                        chunk["chunk_id"],
                        chunk["text"],
                        json.dumps(chunk["headings"], ensure_ascii=False),
                        json.dumps(chunk["pages"]),
                        chunk["source"],
                    ),
                )
                self._conn.execute(
                    "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                    (row_id, sqlite_vec.serialize_float32(vector)),
                )

    def search(self, query_vector: list[float], k: int = 5) -> list[dict]:
        """Return the k chunks closest to the query vector, best first."""
        rows = self._conn.execute(
            """
            WITH knn AS (
                SELECT rowid, distance
                FROM vec_chunks
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT c.chunk_id, c.text, c.headings, c.pages, c.source, knn.distance
            FROM knn JOIN chunks AS c ON c.id = knn.rowid
            ORDER BY knn.distance
            """,
            (sqlite_vec.serialize_float32(query_vector), k),
        ).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "headings": json.loads(row["headings"]),
                "pages": json.loads(row["pages"]),
                "source": row["source"],
                # Vectors are normalized, so cosine similarity = 1 - distance² / 2
                "score": 1 - row["distance"] ** 2 / 2,
            }
            for row in rows
        ]
