import sys
from pathlib import Path

# the model package lives in scripts/undetect
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
