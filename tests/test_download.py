"""download.fetch: whole-file transfer, and refusal when the disk lacks room."""

import collections

import pytest

from drydown_q import download


def test_fetch_copies_the_file(tmp_path):
    src = tmp_path / "server" / "a.nc"
    src.parent.mkdir()
    src.write_bytes(b"x" * 1000)
    size = download.fetch(src.as_uri(), tmp_path / "got" / "a.nc", 1, 0, 0, 64)
    assert size == 1000 and (tmp_path / "got" / "a.nc").read_bytes() == src.read_bytes()


def test_fetch_refuses_when_disk_is_too_small(tmp_path, monkeypatch):
    src = tmp_path / "server" / "a.nc"
    src.parent.mkdir()
    src.write_bytes(b"x" * 1000)
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: usage(1000, 999, 999))
    with pytest.raises(OSError, match="MiB free"):
        download.fetch(src.as_uri(), tmp_path / "got" / "a.nc", 1, 0, 0, 64)
    assert not list((tmp_path / "got").glob("*"))                 # nothing written
