from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta

DB_TYPES = ["postgres","mysql","oracle","mongodb","redis","cassandra"]
REGIONS = ["us-east","us-west","eu","asia"]
STORAGE = ["ssd","hdd","nvme"]

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class SyntheticSpec:
    days: int = 90
    n_dbs: int = 120
    freq_minutes: int = 1
    seed: int = 42

def generate_telemetry(spec: SyntheticSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    start = datetime.utcnow() - timedelta(days=spec.days)
    n_points = int((spec.days * 24 * 60) / spec.freq_minutes)
    ts = np.array([start + timedelta(minutes=i*spec.freq_minutes) for i in range(n_points)], dtype="datetime64[ns]")

    db_ids = [f"db_{i:04d}" for i in range(spec.n_dbs)]
    db_type = rng.choice(DB_TYPES, size=spec.n_dbs, p=[0.25,0.22,0.10,0.18,0.15,0.10])
    region = rng.choice(REGIONS, size=spec.n_dbs, p=[0.45,0.25,0.20,0.10])
    storage_type = rng.choice(STORAGE, size=spec.n_dbs, p=[0.55,0.15,0.30])
    size_gb = np.clip(rng.lognormal(mean=4.0, sigma=0.7, size=spec.n_dbs), 5, 6000)
    archetype = rng.integers(0, 5, size=spec.n_dbs)
    base_alloc = np.clip(rng.normal(loc=4.0, scale=2.0, size=spec.n_dbs), 1, 32)

    rows = []
    for i, db in enumerate(db_ids):
        t = np.arange(n_points)
        daily = (np.sin(2*np.pi*t/(24*60/spec.freq_minutes)) + 1)/2
        weekly = (np.sin(2*np.pi*t/(7*24*60/spec.freq_minutes) + 1.2) + 1)/2

        type_factor = {"postgres":1.0,"mysql":0.9,"oracle":1.15,"mongodb":1.05,"redis":0.7,"cassandra":1.1}[db_type[i]]
        arch_factor = [0.7,1.0,1.3,0.9,1.15][archetype[i]]

        burst = rng.lognormal(mean=-1.2, sigma=0.7, size=n_points)
        qps = (20 + 180*daily*arch_factor + 60*weekly*type_factor) * (0.6 + 0.8*burst)
        qps = np.clip(qps + rng.normal(0, 8, size=n_points), 0, None)

        if archetype[i] in [0,3]:
            read_ratio = np.clip(rng.normal(0.85,0.08,size=n_points), 0.5, 0.99)
        elif archetype[i] in [2]:
            read_ratio = np.clip(rng.normal(0.45,0.10,size=n_points), 0.05, 0.8)
        else:
            read_ratio = np.clip(rng.normal(0.65,0.10,size=n_points), 0.2, 0.95)
        write_ratio = 1.0 - read_ratio

        txn_bytes = np.clip(rng.lognormal(mean=7.5, sigma=0.55, size=n_points), 200, 80000)

        size_factor = np.log1p(size_gb[i]) / np.log(1000)
        req_vcpu = 0.6 + 0.0035*qps + 0.00002*txn_bytes + 0.7*write_ratio*size_factor*type_factor
        req_vcpu *= (0.85 + 0.35*weekly) * (0.9 + 0.2*daily) * arch_factor
        req_vcpu += rng.normal(0, 0.4, size=n_points)
        req_vcpu = np.clip(req_vcpu, 0.5, 64)

        vcpu_allocated = np.clip(base_alloc[i] + rng.normal(0, 0.3, size=n_points), 1, 64)

        util = 100 * _sigmoid(2.8*(req_vcpu/vcpu_allocated - 1.0))
        util = np.clip(util + rng.normal(0, 4, size=n_points), 0, 100)

        p50_latency = 3 + 0.10*qps**0.35 + 0.9*_sigmoid(5*(req_vcpu/vcpu_allocated - 0.95))*60
        p95_latency = p50_latency * (1.6 + 1.5*_sigmoid(4*(req_vcpu/vcpu_allocated - 1.05)))
        p50_latency = np.clip(p50_latency + rng.normal(0,0.8,size=n_points), 0.3, None)
        p95_latency = np.clip(p95_latency + rng.normal(0,1.5,size=n_points), 0.8, None)

        io_read = np.clip((qps*read_ratio)*0.002 + rng.normal(0,0.2,size=n_points), 0, None)
        io_write = np.clip((qps*write_ratio)*0.003 + rng.normal(0,0.25,size=n_points), 0, None)

        mem_util = np.clip(40 + 25*size_factor + 10*weekly + rng.normal(0,5,size=n_points), 5, 98)
        cache_hit = np.clip(0.9 - 0.25*write_ratio + rng.normal(0,0.04,size=n_points), 0.2, 0.99)

        connections = np.clip(10 + 0.18*qps + rng.normal(0,6,size=n_points), 1, None)
        errors = np.clip(rng.poisson(lam=0.03 + 0.08*_sigmoid(6*(req_vcpu/vcpu_allocated - 1.1))), 0, None)

        df = pd.DataFrame({
            "timestamp": ts.astype("datetime64[ns]"),
            "db_id": db,
            "db_type": db_type[i],
            "region": region[i],
            "storage_type": storage_type[i],
            "size_gb": float(size_gb[i]),
            "qps": qps,
            "read_ratio": read_ratio,
            "write_ratio": write_ratio,
            "txn_bytes": txn_bytes,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "errors_per_min": errors.astype(float),
            "connections": connections,
            "cpu_util": util,
            "mem_util": mem_util,
            "io_read_mb_s": io_read,
            "io_write_mb_s": io_write,
            "cache_hit_ratio": cache_hit,
            "vcpu_allocated": vcpu_allocated,
            "true_required_vcpu": req_vcpu,
        })
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out
