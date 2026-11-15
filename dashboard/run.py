"""Start the dashboard with the project's Python 3.12 interpreter."""
import os
from pathlib import Path
import sys


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "This dashboard targets Python 3.12. Run: "
            ".venv-dashboard-py312/bin/python dashboard/run.py"
        )
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.execv(sys.executable, [
        sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
        *sys.argv[1:],
    ])


if __name__ == "__main__":
    main()
