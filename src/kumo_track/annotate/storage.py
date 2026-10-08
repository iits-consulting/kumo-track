"""Storage abstraction for source videos and dataset exports.

All user-data I/O in the app goes through a :class:`Storage` backend so the same
code runs against the local filesystem (dev/tests, today's behaviour) or Azure
Blob Storage (deployment), selected by config. Blob keys are not POSIX paths and
have no ``open()``/seek semantics, so two operations are special:

* **decode** needs a local seekable file — :meth:`Storage.materialize_video`
  returns a real path (the file itself for local; a downloaded scratch copy for
  blob), which :class:`~kumo_track.annotate.frames.FrameSource` (cv2) can seek.
* **export** writes a set of files under a prefix via :class:`ExportWriter`.

The RAM/JPEG frame caches in ``frames.py`` stay local-and-ephemeral — they are
derived, immutable-per-source scratch and never go through this layer.
"""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class ExportWriter(ABC):
    """A sink for one export's files, addressed by relative path under a prefix."""

    @abstractmethod
    def write_bytes(self, rel_path: str, data: bytes) -> None:
        """Write one file (e.g. ``images/foo.jpg`` or ``labels.json``)."""

    @property
    @abstractmethod
    def location(self) -> str:
        """Human-readable location of the export (dir path or blob prefix)."""


class Storage(ABC):
    """Backend for source videos + exports (local FS or Azure Blob)."""

    @property
    @abstractmethod
    def location(self) -> str:
        """Human-readable location clips are served from (dir path or blob container)."""

    @abstractmethod
    def list_videos(self) -> list[str]:
        """Sorted basenames of available source clips."""

    @abstractmethod
    def video_exists(self, name: str) -> bool: ...

    @abstractmethod
    def save_video(self, name: str, fileobj: BinaryIO) -> None:
        """Persist an uploaded clip from a binary file-like."""

    @abstractmethod
    def materialize_video(self, name: str) -> Path:
        """Return a local seekable path for ``name`` (for cv2 decode)."""

    @abstractmethod
    def export_writer(self, out_name: str) -> ExportWriter: ...


# --- local filesystem ----------------------------------------------------------


class _LocalExportWriter(ExportWriter):
    def __init__(self, root: Path):
        self._root = root

    def write_bytes(self, rel_path: str, data: bytes) -> None:
        dest = self._root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    @property
    def location(self) -> str:
        return str(self._root)


class LocalStorage(Storage):
    """Today's behaviour: videos in ``video_dir``, exports under ``export_root``.

    ``export_root`` is kept relative (default ``data``) so it resolves against the
    cwd at write time, matching the pre-abstraction export path."""

    def __init__(self, video_dir: Path, export_root: Path = Path("data")):
        self.video_dir = Path(video_dir)
        self.export_root = Path(export_root)

    @property
    def location(self) -> str:
        return str(self.video_dir)

    def list_videos(self) -> list[str]:
        if not self.video_dir.exists():
            return []
        return sorted(
            p.name for p in self.video_dir.glob("*") if p.suffix.lower() in VIDEO_EXTS
        )

    def video_exists(self, name: str) -> bool:
        return (self.video_dir / name).exists()

    def save_video(self, name: str, fileobj: BinaryIO) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)
        with (self.video_dir / name).open("wb") as out:
            shutil.copyfileobj(fileobj, out, length=1 << 20)

    def materialize_video(self, name: str) -> Path:
        return self.video_dir / name

    def export_writer(self, out_name: str) -> ExportWriter:
        return _LocalExportWriter(self.export_root / out_name)


# --- azure blob ----------------------------------------------------------------


class _BlobExportWriter(ExportWriter):
    def __init__(self, container_client, prefix: str):
        self._container = container_client
        self._prefix = prefix

    def write_bytes(self, rel_path: str, data: bytes) -> None:
        self._container.upload_blob(f"{self._prefix}/{rel_path}", data, overwrite=True)

    @property
    def location(self) -> str:
        return f"{self._container.container_name}/{self._prefix}"


class BlobStorage(Storage):
    """Azure Blob backend. Videos and exports live in (possibly the same) blob
    containers; source clips are downloaded to an ephemeral scratch dir for decode.
    """

    def __init__(
        self,
        connection_string: str,
        video_container: str,
        export_container: str,
        scratch_dir: Path,
    ):
        from azure.storage.blob import BlobServiceClient

        svc = BlobServiceClient.from_connection_string(connection_string)
        self._videos = svc.get_container_client(video_container)
        self._exports = svc.get_container_client(export_container)
        self._scratch = Path(scratch_dir)
        self._scratch.mkdir(parents=True, exist_ok=True)

    @property
    def location(self) -> str:
        return self._videos.container_name

    def list_videos(self) -> list[str]:
        # Root-level clips only: skip any nested key (e.g. derived/<name>.preview.mp4),
        # which are thumbnails/previews, not real samples.
        return sorted(
            b.name for b in self._videos.list_blobs()
            if "/" not in b.name and Path(b.name).suffix.lower() in VIDEO_EXTS
        )

    def video_exists(self, name: str) -> bool:
        return self._videos.get_blob_client(name).exists()

    def save_video(self, name: str, fileobj: BinaryIO) -> None:
        self._videos.upload_blob(name, fileobj, overwrite=True)

    def materialize_video(self, name: str) -> Path:
        # Cache by name in scratch so reopening a clip doesn't re-download. Source
        # blobs are immutable per name (upload sanitises to a basename), so a
        # present local copy is always valid.
        local = self._scratch / name
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            # Download to a temp name UNIQUE to this call, then atomically rename.
            # Two users opening the same not-yet-cached clip run materialize
            # concurrently (open_video's cache check misses for both), so a shared
            # ``.part`` would be corrupted — the second writer's open("wb") truncates
            # it mid-download of the first. A per-call name means each writes its own
            # complete file and the last os.replace() wins; no reader sees a partial.
            tmp = local.with_suffix(f"{local.suffix}.{os.getpid()}-{os.urandom(4).hex()}.part")
            try:
                with tmp.open("wb") as out:
                    self._videos.download_blob(name).readinto(out)
                os.replace(tmp, local)
            finally:
                tmp.unlink(missing_ok=True)  # clean up on error (no-op after replace)
        return local

    def export_writer(self, out_name: str) -> ExportWriter:
        return _BlobExportWriter(self._exports, out_name)


# --- selection -----------------------------------------------------------------


def make_storage(video_dir: Path) -> Storage:
    """Pick a backend from the environment.

    ``STORAGE_BACKEND=azure`` (or an ``AZURE_STORAGE_CONNECTION_STRING`` being set
    with the backend unset) selects Azure Blob; otherwise the local filesystem
    under ``video_dir`` (the default used by dev and the test suite).
    """
    backend = os.environ.get("STORAGE_BACKEND", "").lower()
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if backend == "azure" or (backend == "" and conn_str):
        if not conn_str:
            raise RuntimeError("STORAGE_BACKEND=azure requires AZURE_STORAGE_CONNECTION_STRING")
        return BlobStorage(
            connection_string=conn_str,
            video_container=os.environ.get("AZURE_VIDEO_CONTAINER", "videos"),
            export_container=os.environ.get("AZURE_EXPORT_CONTAINER", "exports"),
            scratch_dir=Path(os.environ.get("SCRATCH_DIR", "/tmp/kumo-track/videos")),
        )
    return LocalStorage(video_dir)
