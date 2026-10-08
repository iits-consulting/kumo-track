"""SAM3-assisted video annotation: SQLite store + tracker session.

The FastAPI app lives in ``scripts/annotate_app.py`` (the project's entrypoint
convention); the reusable logic — the annotation store and the SAM3 video
tracker wrapper — lives here so it can be imported and tested directly.
"""
