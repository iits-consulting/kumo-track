"""Entrypoint for the SAM3 video annotation app.

Run with:  poe annotate   (==  uv run python scripts/annotate_app.py)

All app logic lives in :mod:`kumo_track.annotate.app`; this script just wires up
the import path, loads ``.env``, and serves ``create_app()``.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

import uvicorn

from kumo_track.annotate.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
