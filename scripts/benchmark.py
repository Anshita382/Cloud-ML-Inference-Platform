#!/usr/bin/env python3
"""
Benchmark script — compares performance across batch configurations.

Runs multiple Locust sessions with different batch sizes and reports
a comparison table of latency and throughput metrics.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --users 500 --duration 60
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_locust(
    host: str,
    users: int,
    spawn_rate: int,
    duration: int,
    report_name: str,
) -> dict:
    """Run a single Locust test and return stats."""
    csv_prefix = f"/tmp/locust_{report_name}"

    cmd = [
        "locust",
        "-f", "loadtest/locustfile.py",
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", f"{duration}s",
        "--headless",
        "--csv", csv_prefix,
        "--only-summary",
    ]

    print(f"\n{'='*60}")
    print(f"Running: {report_name}")
    print(f"Users: {users}, Duration: {duration}s")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)

    if result.returncode != 0:
        print(f"Locust failed: {result.stderr}")
        return {}

    # Parse CSV stats
    stats_file = f"{csv_prefix}_stats.csv"
    if not os.path.exists(stats_file):
        print(f"Stats file not found: {stats_file}")
        return {}

    import csv
    stats = {}
    with open(stats_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name") == "Aggregated":
                stats = {
                    "rps": float(row.get("Requests/s", 0)),
                    "p50_ms": float(row.get("50%", 0)),
                    "p95_ms": float(row.get("95%", 0)),
                    "p99_ms": float(row.get("99%", 0)),
                    "avg_ms": float(row.get("Average Response Time", 0)),
                    "failures": int(row.get("Failure Count", 0)),
                    "total": int(row.get("Request Count", 0)),
                }
                break

    return stats


def print_comparison(results: list[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(
        f"{'Config':<25} {'RPS':>8} {'p50 (ms)':>10} {'p95 (ms)':>10} "
        f"{'p99 (ms)':>10} {'Errors':>8} {'Total':>8}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r['name']:<25} {r['stats'].get('rps', 0):>8.1f} "
            f"{r['stats'].get('p50_ms', 0):>10.0f} "
            f"{r['stats'].get('p95_ms', 0):>10.0f} "
            f"{r['stats'].get('p99_ms', 0):>10.0f} "
            f"{r['stats'].get('failures', 0):>8} "
            f"{r['stats'].get('total', 0):>8}"
        )

    print(f"{'='*80}")

    # Save to file
    report_path = Path("benchmark_results.json")
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nResults saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="ML Inference Benchmark")
    parser.add_argument("--host", default="http://localhost:8000", help="API host")
    parser.add_argument("--users", type=int, default=100, help="Concurrent users")
    parser.add_argument("--spawn-rate", type=int, default=20, help="User spawn rate")
    parser.add_argument("--duration", type=int, default=30, help="Test duration (seconds)")
    args = parser.parse_args()

    configs = [
        {"name": "light-load-50-users", "users": 50},
        {"name": "medium-load-100-users", "users": 100},
        {"name": "heavy-load-500-users", "users": 500},
        {"name": "burst-load-1000-users", "users": 1000},
    ]

    results = []
    for config in configs:
        users = min(config["users"], args.users * 10)  # Cap at 10x requested
        stats = run_locust(
            host=args.host,
            users=users,
            spawn_rate=args.spawn_rate,
            duration=args.duration,
            report_name=config["name"],
        )
        results.append({"name": config["name"], "stats": stats})
        time.sleep(5)  # Cool down between tests

    print_comparison(results)


if __name__ == "__main__":
    main()
