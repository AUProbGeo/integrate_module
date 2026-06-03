import subprocess
import sys
from pathlib import Path


def main():
    app = Path(__file__).parent.parent / "streamlit" / "integrate_www.py"
    sys.exit(subprocess.run(["streamlit", "run", str(app)] + sys.argv[1:]).returncode)
