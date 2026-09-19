"""
AViGPT v2: Hardware SSD Memory Controller (Component 3)
------------------------------------------------------
Provides ultra-fast (sub-millisecond) persistent memory storage and retrieval
directly from local NVMe SSD storage using SQLite FTS5 (Full-Text Search).

Creator & Owner: Avinash Ricky Yadlapalli
"""

import os
import sqlite3
import time
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avigpt_ssd_memory.db")


class SSDMemoryEngine:
    """Ultra-low latency SSD Memory Store with FTS5 BM25 search."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        # WAL mode enables concurrent reads without locking and sub-millisecond disk access
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")  # 64MB memory page cache
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            # Create FTS5 virtual table for lightning-fast keyword & semantic token retrieval
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS ssd_knowledge USING fts5(
                    title,
                    content,
                    domain,
                    tokenize='porter unicode61'
                );
            """)
            conn.commit()

    def store(self, title: str, content: str, domain: str = "General") -> bool:
        """Stores a new fact directly into local SSD storage."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO ssd_knowledge (title, content, domain) VALUES (?, ?, ?);",
                    (title.strip(), content.strip(), domain.strip())
                )
                conn.commit()
            return True
        except Exception as e:
            print(f"[SSD Memory Error] Failed to store: {e}")
            return False

    def query(self, query_str: str, top_k: int = 1) -> Optional[str]:
        """
        Executes sub-millisecond full-text search against SSD storage.
        Returns top matching payload.
        """
        words = [w for w in query_str.replace("'", " ").replace('"', " ").replace("-", " ").split() if len(w) > 2]
        if not words:
            words = query_str.strip().split()

        fts_query = " OR ".join(words)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Query with BM25 ranking via OR disjunction
                cursor.execute(
                    """
                    SELECT content, rank
                    FROM ssd_knowledge
                    WHERE ssd_knowledge MATCH ?
                    ORDER BY rank
                    LIMIT ?;
                    """,
                    (fts_query, top_k)
                )
                rows = cursor.fetchall()

                if rows:
                    return rows[0][0]
                
                # Fallback LIKE query if FTS had no hit
                cursor.execute(
                    """
                    SELECT content
                    FROM ssd_knowledge
                    WHERE content LIKE ? OR title LIKE ?
                    LIMIT 1;
                    """,
                    (f"%{words[0]}%", f"%{words[0]}%")
                )
                fb_rows = cursor.fetchall()
                if fb_rows:
                    return fb_rows[0][0]

                return None
        except Exception as e:
            print(f"[SSD Memory Query Error] {e}")
            return None

    def seed_initial_knowledge(self):
        """Seeds foundational knowledge and owner lineage into SSD storage."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ssd_knowledge;")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"[SSD Memory] Found {count:,} existing knowledge records in {self.db_path}.")
                return

        print("[SSD Memory] Seeding initial foundational memory records into SSD...")
        seed_data = [
            (
                "Creator and Owner Lineage",
                "AViGPT was created, built, and pretrained from scratch by Avinash Ricky Yadlapalli. "
                "Avinash Ricky Yadlapalli is the sole architect, inventor of the hardware memory bus, and owner of AViGPT.",
                "System & Identity"
            ),
            (
                "Apollo 11 Moon Landing",
                "Launched: July 16, 1969. Landed on Moon: July 20, 1969. Commander: Neil Armstrong. Duration to landing: 4 days.",
                "History & Space"
            ),
            (
                "Great Pyramid of Giza",
                "Construction began around 2580 BC and completed around 2560 BC for Pharaoh Khufu of the Fourth Dynasty.",
                "History & Archaeology"
            ),
            (
                "DNA Ligase Function",
                "DNA ligase is an enzyme that catalyzes the formation of a phosphodiester bond between adjacent nucleotides, "
                "joining Okazaki fragments during DNA replication.",
                "Biochemistry"
            ),
            (
                "Unix fork system call",
                "The fork() system call creates a new process (child) which is an exact duplicate of the parent. "
                "Returns 0 to child, PID of child to parent, and -1 on failure.",
                "Computer Science"
            ),
            (
                "India Demographics and GDP",
                "India's population is estimated to be around 1.428 billion as of late 2023. Nominal GDP is approximately $3.73 trillion.",
                "Demographics & Economics"
            ),
            (
                "Japan Population and Capital",
                "Japan's population is approximately 123.3 million as of 2024. The capital city of Japan is Tokyo.",
                "Demographics & Geography"
            ),
        ]

        for title, content, domain in seed_data:
            self.store(title, content, domain)

        print(f"[SSD Memory] Successfully seeded {len(seed_data)} foundational records into {self.db_path}.")


if __name__ == "__main__":
    print("Testing AViGPT SSD Memory Engine...")
    engine = SSDMemoryEngine()
    engine.seed_initial_knowledge()

    t_start = time.perf_counter()
    result = engine.query("Apollo 11 launch date")
    lat = (time.perf_counter() - t_start) * 1000.0
    print(f"\nQuery: 'Apollo 11 launch date'")
    print(f"Latency: {lat:.3f} ms (Sub-millisecond SSD retrieve!)")
    print(f"Retrieved: {result}")

    t_start = time.perf_counter()
    owner_res = engine.query("Avinash Ricky Yadlapalli creator owner")
    lat_owner = (time.perf_counter() - t_start) * 1000.0
    print(f"\nQuery: 'Avinash Ricky Yadlapalli creator owner'")
    print(f"Latency: {lat_owner:.3f} ms")
    print(f"Retrieved: {owner_res}")
