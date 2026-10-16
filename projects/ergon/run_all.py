"""
Start every employee in employees/ in parallel.
Usage: python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from shared.employees import load_all


def run_employees():
    employees = load_all()  # validates every YAML first: unique names and ports
    processes = []

    print("Starting all employees...")
    print("-" * 40)

    for emp in employees:
        process = subprocess.Popen(
            [sys.executable, "-m", "shared.employee_server", emp.id],
            cwd=str(BASE_DIR),
        )
        processes.append(process)
        print(f"[OK] {emp.name} started on port {emp.port}")

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
                    print(f"Warning: {employees[i].name} stopped unexpectedly (exit code {proc.returncode})")
    except KeyboardInterrupt:
        print("\nStopping all employees...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()
        print("All employees stopped.")


if __name__ == "__main__":
    run_employees()
