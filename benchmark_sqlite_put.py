import sqlite3
import time
import json
import datetime

def setup_db(conn):
    conn.execute("""
CREATE TABLE IF NOT EXISTS store (
    prefix text NOT NULL,
    key text NOT NULL,
    value text NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    ttl_minutes REAL,
    PRIMARY KEY (prefix, key)
);
""")
    conn.commit()

def benchmark_put_ops(conn, n_ops=1000):
    setup_db(conn)

    # Simulate data for _prepare_batch_PUT_queries
    # (prefix, key, value, expires_at, ttl)
    inserts = [
        (("ns",), f"key_{i}", {"data": f"val_{i}"}, None, None)
        for i in range(n_ops)
    ]

    # Replicate _prepare_batch_PUT_queries logic for inserts
    values = []
    insertion_params = []
    for ns, key, value, expires_at, ttl in inserts:
        values.append("(?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)")
        insertion_params.extend([
            ".".join(ns),
            key,
            json.dumps(value),
            expires_at,
            ttl
        ])

    values_str = ",".join(values)
    query = f"""
        INSERT OR REPLACE INTO store (prefix, key, value, created_at, updated_at, expires_at, ttl_minutes)
        VALUES {values_str}
    """

    # Baseline: Current execute
    start = time.perf_counter()
    cur = conn.cursor()
    conn.execute("BEGIN")
    cur.execute(query, insertion_params)
    conn.execute("COMMIT")
    end = time.perf_counter()
    baseline_time = end - start
    print(f"Sync Execute (Big Query): {baseline_time:.4f}s")

    # Clear
    conn.execute("DELETE FROM store")
    conn.commit()

    # Alternative: executemany
    # We need a fixed query and list of params
    single_insert_query = """
        INSERT OR REPLACE INTO store (prefix, key, value, created_at, updated_at, expires_at, ttl_minutes)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
    """
    executemany_params = [
        (".".join(ns), key, json.dumps(value), expires_at, ttl)
        for ns, key, value, expires_at, ttl in inserts
    ]

    start = time.perf_counter()
    cur = conn.cursor()
    conn.execute("BEGIN")
    cur.executemany(single_insert_query, executemany_params)
    conn.execute("COMMIT")
    end = time.perf_counter()
    optimized_time = end - start
    print(f"Sync executemany: {optimized_time:.4f}s")
    print(f"Improvement over Big Query: {(baseline_time - optimized_time) / baseline_time * 100:.2f}%")

if __name__ == "__main__":
    conn = sqlite3.connect(":memory:")
    benchmark_put_ops(conn)
