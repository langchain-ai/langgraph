import sqlite3
import time

def setup_db(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS store (prefix TEXT, key TEXT, value TEXT, PRIMARY KEY (prefix, key))")
    conn.commit()

def benchmark_sync(conn, n=1000):
    setup_db(conn)
    queries = [
        ("INSERT OR REPLACE INTO store (prefix, key, value) VALUES (?, ?, ?)", ("ns", f"key_{i}", "val"))
        for i in range(n)
    ]

    # Baseline: Current loop
    start = time.perf_counter()
    cur = conn.cursor()
    conn.execute("BEGIN")
    for query, params in queries:
        cur.execute(query, params)
    conn.execute("COMMIT")
    end = time.perf_counter()
    baseline_time = end - start
    print(f"Sync Loop: {baseline_time:.4f}s")

    # Optimized: executemany
    # We need to clear data first
    conn.execute("DELETE FROM store")
    conn.commit()

    start = time.perf_counter()
    cur = conn.cursor()
    conn.execute("BEGIN")
    # In the actual code, queries might have different SQL.
    # But if they are same, we can group them.
    # For this benchmark, we assume same SQL.
    cur.executemany(queries[0][0], [p for q, p in queries])
    conn.execute("COMMIT")
    end = time.perf_counter()
    optimized_time = end - start
    print(f"Sync executemany: {optimized_time:.4f}s")
    print(f"Improvement: {(baseline_time - optimized_time) / baseline_time * 100:.2f}%")

if __name__ == "__main__":
    conn = sqlite3.connect(":memory:")
    benchmark_sync(conn)
