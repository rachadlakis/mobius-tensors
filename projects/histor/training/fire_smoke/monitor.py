"""Live page for the fire/smoke training runs: laptop runs from runs/, pod runs over SSH.

    .venv/Scripts/python.exe -I monitor.py --pod root@216.81.151.71:14503 \
        --pod-start 2026-10-15T06:09:29Z --rate 0.72
    then open http://localhost:8765

Every minute it reads runs/*/results.csv + status.json here and on the pod (one SSH call), and
the page redraws. When the pod is unreachable (stopped) the last data stays on screen.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLS = {"loss": "train/loss", "map50": "metrics/mAP50", "map": "metrics/mAP50-95", "recall": "metrics/recall"}
STATE: dict = {"runs": {}, "pod": {"ok": None, "checked": None}}


def parse_run(status_text: str, csv_text: str) -> dict:
    try:
        status = json.loads(status_text) if status_text.strip() else {}
    except json.JSONDecodeError:
        status = {}
    rows = []
    for r in csv.DictReader(io.StringIO(csv_text)):
        try:
            rows.append({"epoch": int(float(r["epoch"])), **{k: float(r[c]) for k, c in COLS.items() if r.get(c)}})
        except (KeyError, ValueError):
            continue
    return {
        "state": status.get("state", "unknown"),
        "epoch": status.get("completed_epochs", rows[-1]["epoch"] if rows else 0),
        "total": status.get("total_epochs") or status.get("epochs"),
        "eta_s": status.get("eta_seconds"),
        "rows": rows,
    }


def read_local() -> dict:
    runs = {}
    for d in sorted((HERE / "runs").glob("*/")):
        if (d / "results.csv").exists() and d.name != "trial_indoor":
            st = d / "status.json"
            run = parse_run(st.read_text(encoding="utf-8") if st.exists() else "", (d / "results.csv").read_text(encoding="utf-8"))
            run["where"] = "laptop" if "runpod" not in d.name and d.name != "yolo9s_640" else "pod (copy)"
            run["updated"] = (d / "results.csv").stat().st_mtime
            runs[d.name] = run
    return runs


def read_pod(target: str, root: str) -> dict | None:
    host, _, port = target.partition(":")
    cmd = (
        f"for d in {root}/runs/*/; do [ -f $d/results.csv ] || continue; echo \"@@RUN $(basename $d)\"; "
        f"echo @@STATUS; cat $d/status.json 2>/dev/null; echo; echo @@CSV; cat $d/results.csv; "
        f"echo @@MTIME $(stat -c %Y $d/results.csv); done"
    )
    try:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "-p", port or "22", host, cmd],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    runs = {}
    for block in out.stdout.split("@@RUN ")[1:]:
        name, rest = block.split("\n", 1)
        status, rest = rest.split("@@CSV\n", 1)
        csv_text, _, mtime = rest.rpartition("@@MTIME ")
        run = parse_run(status.replace("@@STATUS\n", ""), csv_text)
        run["where"] = "pod"
        run["updated"] = float(mtime.strip() or 0)
        runs[name.strip()] = run
    return runs


def refresh_loop(args) -> None:
    pod_runs: dict = {}
    while True:
        if args.pod:
            got = read_pod(args.pod, args.pod_root)
            STATE["pod"] = {"ok": got is not None, "checked": time.time()}
            if got is not None:
                pod_runs = got
        runs = read_local()
        runs.update(pod_runs)  # the pod's live copy wins over an old local one
        STATE["runs"] = runs
        if args.pod_start:
            start = datetime.fromisoformat(args.pod_start.replace("Z", "+00:00"))
            hours = (datetime.now(timezone.utc) - start).total_seconds() / 3600
            STATE["pod"].update(hours=round(hours, 2), cost=round(hours * args.rate, 2), rate=args.rate)
        time.sleep(args.every)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/data.json"):
            body, kind = json.dumps(STATE).encode(), "application/json"
        else:
            body, kind = (HERE / "monitor.html").read_bytes(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a) -> None:  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pod", default="", help="user@host:port of the pod's direct SSH")
    ap.add_argument("--pod-root", default="/root/fire_smoke")
    ap.add_argument("--pod-start", default="", help="pod start time (UTC, ISO) for the cost line")
    ap.add_argument("--rate", type=float, default=0.0, help="pod $ per hour")
    ap.add_argument("--every", type=int, default=60, help="seconds between refreshes")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    threading.Thread(target=refresh_loop, args=(args,), daemon=True).start()
    print(f"monitor on http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
