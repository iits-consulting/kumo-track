"""Postgres backend smoke test (FIX-3).

The production metadata backend is PostgreSQL, but the rest of the suite runs on
SQLite. This module runs the core DB assertions against a *real* Postgres so the
dialect-specific paths — the ``?``→``%s`` translation, ``RETURNING id``, the
identity PK, ``ON CONFLICT DO UPDATE``, the window-function query, and pool
checkout — are actually exercised.

It selects a server in this order and **skips cleanly** if none is available, so
``uv run pytest`` on a laptop without Postgres stays green:
  1. ``KUMO_TEST_POSTGRES_URL`` env var (a CI service container / shared server);
  2. an ephemeral testcontainers-python Postgres, if the package + Docker exist.
"""

import os

import pytest

from kumo_track.annotate import db


@pytest.fixture(scope="module")
def pg_url():
    env_url = os.environ.get("KUMO_TEST_POSTGRES_URL")
    if env_url:
        yield env_url
        return
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("no Postgres: set KUMO_TEST_POSTGRES_URL or install testcontainers")
    try:
        with PostgresContainer("postgres:16-alpine") as pg:
            # psycopg (v3) uses the bare postgresql:// scheme, not +psycopg2.
            yield pg.get_connection_url().replace("+psycopg2", "")
    except Exception as exc:  # Docker not running / image pull failed
        pytest.skip(f"could not start a Postgres testcontainer: {exc}")


@pytest.fixture
def pg(pg_url):
    """A clean-schema connection to the test Postgres (tables dropped each test)."""
    pytest.importorskip("psycopg")
    conn = db.connect(pg_url)  # ensures the PG schema exists
    conn.execute("DROP TABLE IF EXISTS annotation, object, video CASCADE")
    conn.commit()
    db._ensure_schema_pg(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_pg_is_selected(pg):
    assert pg.is_postgres is True


def test_pg_insert_returns_id(pg):
    # RETURNING id path (db.py _insert) + the ?→%s translation.
    vid = db.upsert_video(pg, "v.mp4", "/tmp/v.mp4", 100, 80, 10.0, 4, None, 10, list(range(0, 40, 4)))
    oid = db.create_object(pg, vid, "bag", "alice")
    assert isinstance(vid, int) and isinstance(oid, int)
    assert db.object_video_id(pg, oid) == vid


def test_pg_upsert_annotation_is_last_writer_wins(pg):
    # ON CONFLICT (object_id, frame_idx) DO UPDATE on the real backend.
    vid = db.upsert_video(pg, "v.mp4", "/tmp/v.mp4", 100, 80, 10.0, 4, None, 10, list(range(0, 40, 4)))
    oid = db.create_object(pg, vid, "bag", "alice")
    first = [[1, 1], [9, 1], [9, 9], [1, 9]]
    second = [[2, 2], [8, 2], [8, 8], [2, 8]]
    db.upsert_annotation(pg, vid, oid, 3, 12, first, None, None, "seed")
    db.upsert_annotation(pg, vid, oid, 3, 12, second, None, None, "manual")
    ann = db.get_annotations(pg, vid)
    assert ann["frames"][3][oid]["corners"] == second
    assert ann["frames"][3][oid]["origin"] == "manual"


def test_pg_get_annotations_roundtrip_via_pool(pg_url):
    # Exercise the Database pool checkout path used by the app (not the single
    # connect() connection), against Postgres.
    pytest.importorskip("psycopg")
    database = db.Database(pg_url)
    try:
        with database.connection() as c:
            c.execute("DROP TABLE IF EXISTS annotation, object, video CASCADE")
            c.commit()
            db._ensure_schema_pg(c)
            vid = db.upsert_video(c, "p.mp4", "/tmp/p.mp4", 64, 48, 10.0, 4, None, 10, list(range(0, 40, 4)))
            oid = db.create_object(c, vid, "zone", "bob")
            box = [[0, 0], [5, 0], [5, 5], [0, 5]]
            db.upsert_annotation(c, vid, oid, 2, 8, box, box, 0.9, "seed")
        with database.connection() as c:  # a fresh checkout sees the committed row
            ann = db.get_annotations(c, vid)
        assert [o["id"] for o in ann["objects"]] == [oid]
        assert ann["objects"][0]["created_by"] == "bob"
        assert ann["frames"][2][oid]["corners"] == box
    finally:
        database.close()
