import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.executor import BaseA2AExecutor


class DataAnalystExecutor(BaseA2AExecutor):
    """Runs the Data Analyst ADK agent behind the A2A protocol (see shared/executor.py)."""
