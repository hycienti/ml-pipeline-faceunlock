"""Flood the app with Locust at 1, 2 and 3 Docker containers and compare latency and throughput.

Needs Docker and locust (pip install -r requirements.txt). From the repo root:

    python scripts/flood_experiment.py                         # 60 users, 60 s per setup
    python scripts/flood_experiment.py --users 50 --seconds 90 --containers 1 2 4

For each container count it scales the compose stack, restarts the load balancer so it sees every
container, runs locustfile.py headless, and reads Locust's CSV. Results go to results/flood/.
"""
import argparse
import csv
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(*cmd: str) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def wait_until_up(host: str, timeout: int = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{host}/login", timeout=5)
            return
        except Exception:
            time.sleep(2)
    sys.exit(f"{host} did not come up within {timeout}s")


def read_stats(path: Path) -> dict[str, dict]:
    with open(path, newline="") as handle:
        return {row["Name"]: row for row in csv.DictReader(handle)}


def run_setup(containers: int, args) -> dict:
    sh("docker", "compose", "up", "-d", "--scale", f"web={containers}", "--wait")
    sh("docker", "compose", "restart", "proxy")  # nginx looks the containers up when it starts
    wait_until_up(args.host)
    prefix = args.out / f"containers_{containers}"
    sh(
        "locust", "-f", "locustfile.py", "--headless", "-u", str(args.users), "-r", str(args.rate),
        "-t", f"{args.seconds}s", "--host", args.host, "--csv", str(prefix), "--reset-stats", "--only-summary",
    )
    stats = read_stats(Path(f"{prefix}_stats.csv"))
    total, predict = stats["Aggregated"], stats["predict (registered face)"]
    return {
        "containers": containers,
        "requests": int(total["Request Count"]),
        "failures": int(total["Failure Count"]),
        "rps": float(total["Requests/s"]),
        "median": float(total["50%"]),
        "p95": float(total["95%"]),
        "p99": float(total["99%"]),
        "predict_median": float(predict["50%"]),
        "predict_p95": float(predict["95%"]),
    }


def write_report(rows: list[dict], args) -> None:
    lines = [
        "# Flood simulation: latency with different numbers of Docker containers",
        "",
        f"Locust, {args.users} simulated users (spawn rate {args.rate}/s), {args.seconds} s per setup, statistics reset after",
        "the ramp-up. Each container is limited to 2 CPUs and runs the full app and model; nginx balances by",
        "least connections. Every user logs in once, then predicts a registered face (weight 4), predicts a",
        "stranger (2), logs in again (1) and loads the status API (1). Times are milliseconds.",
        "",
        "| Containers | Requests | Failures | Requests/s | Median | 95th pct | 99th pct | Predict median | Predict 95th pct |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['containers']} | {r['requests']} | {r['failures']} | {r['rps']:.1f} | {r['median']:.0f} | "
            f"{r['p95']:.0f} | {r['p99']:.0f} | {r['predict_median']:.0f} | {r['predict_p95']:.0f} |"
        )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(r["containers"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, [r["rps"] for r in rows], color="#4cc2a0")
    axes[0].set(title="Throughput", xlabel="Docker containers", ylabel="requests per second")
    width = 0.35
    positions = range(len(rows))
    axes[1].bar([p - width / 2 for p in positions], [r["median"] for r in rows], width, label="median", color="#6aa8ff")
    axes[1].bar([p + width / 2 for p in positions], [r["p95"] for r in rows], width, label="95th percentile", color="#f0b45a")
    axes[1].set_xticks(list(positions), labels)
    axes[1].set(title="Response time", xlabel="Docker containers", ylabel="milliseconds")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.out / "latency.png", dpi=130)
    print((args.out / "summary.md").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--containers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--rate", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--host", default="http://localhost:8080")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "flood")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sh("docker", "compose", "down", "-v")  # start from an empty database every time
    try:
        rows = [run_setup(n, args) for n in args.containers]
    finally:
        sh("docker", "compose", "down", "-v")
    write_report(rows, args)


if __name__ == "__main__":
    main()
