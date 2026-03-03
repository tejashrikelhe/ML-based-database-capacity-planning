# Telemetry Schema (data/raw/telemetry.csv)

Each row is a 1-minute sample for a database instance.

Identity: timestamp, db_id, db_type, region, storage_type, size_gb
Workload: qps, read_ratio, write_ratio, txn_bytes
Perf: p50_latency_ms, p95_latency_ms, errors_per_min, connections
Resources: cpu_util, mem_util, io_read_mb_s, io_write_mb_s, cache_hit_ratio
Current allocation: vcpu_allocated
