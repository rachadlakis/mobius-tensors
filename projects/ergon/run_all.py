"""
Start every employee in parallel.
Usage: python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from shared.config import DATA_PORT, WEB_INTEL_PORT

AGENTS = [
    {"name": "Researcher", "dir": "web_intel_agent_adk", "port": WEB_INTEL_PORT},
    {"name": "Data Analyst", "dir": "data_agent_adk", "port": DATA_PORT},
]


def run_agents():
    processes = []

    print("Starting all employees...")
    print("-" * 40)

    for agent in AGENTS:
        agent_dir = BASE_DIR / agent["dir"]
        main_file = agent_dir / "__main__.py"

        if not main_file.exists():
            print(f"[MISSING] {agent['name']}: __main__.py not found")
            continue

        # Start the agent process
        process = subprocess.Popen(
            [sys.executable, str(main_file)],
            cwd=str(agent_dir),
        )
        processes.append(process)
        print(f"[OK] {agent['name']} started on port {agent['port']}")

    print("-" * 40)
    print(f"All {len(processes)} employees running!")
    print("Press Ctrl+C to stop all employees")
    print("-" * 40)

    try:
        # Keep running until interrupted
        reported_dead = set()
        while True:
            time.sleep(1)
            # Check if any process died (report each one only once)
            for i, proc in enumerate(processes):
                if proc.poll() is not None and i not in reported_dead:
                    reported_dead.add(i)
                    print(f"Warning: {AGENTS[i]['name']} stopped unexpectedly (exit code {proc.returncode})")
    except KeyboardInterrupt:
        print("\nStopping all employees...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()
        print("All employees stopped.")


if __name__ == "__main__":
    run_agents()
