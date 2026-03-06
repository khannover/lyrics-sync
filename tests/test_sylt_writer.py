"""Unit tests for app.sylt_writer.write_sylt_tag embed_mode parameter."""

import os
import tempfile

import pytest
from mutagen.id3 import ID3, SYLT, USLT, TXXX, Encoding
from mutagen.mp3 import MP3

# Minimal silent MP3 bytes with enough valid MPEG frames for mutagen.mp3.MP3() to parse.
# Each MPEG1 Layer3 128kbps 44100Hz frame is 417 bytes (4-byte header + 413-byte payload).
# mutagen requires at least 4 valid frames.
_MPEG_FRAME = bytes([0xFF, 0xFB, 0x90, 0x00]) + bytes(413)
_SILENT_MP3_BYTES = (
    # ID3v2.3 header: "ID3" + version 2.3 + flags 0 + syncsafe size 0
    bytes([0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    + _MPEG_FRAME * 4
)

_SYNCED_LYRICS = [("Hello", 1000), ("World", 5000)]
_PLAIN_LYRICS = "Plain unsynced lyrics"


def _make_mp3(tmp_path: str) -> str:
    path = os.path.join(tmp_path, "test.mp3")
    with open(path, "wb") as f:
        f.write(_SILENT_MP3_BYTES)
    return path


def _add_plain_lyrics(path: str):
    """Add pre-existing plain (unsynced) USLT and TXXX:LYRICS tags."""
    try:
        tags = ID3(path)
    except Exception:
        tags = ID3()
    tags.add(USLT(encoding=Encoding.UTF8, lang="eng", desc="", text=_PLAIN_LYRICS))
    tags.add(TXXX(encoding=Encoding.UTF8, desc="LYRICS", text=[_PLAIN_LYRICS]))
    tags.save(path)


from app.sylt_writer import write_sylt_tag


class TestWriteSyltTagOverwrite:
    def test_overwrite_writes_sylt_uslt_txxx(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            write_sylt_tag(path, _SYNCED_LYRICS, embed_mode="overwrite")
            tags = ID3(path)

            sylt_keys = [k for k in tags if k.startswith("SYLT")]
            uslt_keys = [k for k in tags if k.startswith("USLT")]
            txxx_keys = [k for k in tags if k.startswith("TXXX:LYRICS")]

            assert sylt_keys, "SYLT frame should be written"
            assert uslt_keys, "USLT frame should be written"
            assert txxx_keys, "TXXX:LYRICS frame should be written"

    def test_overwrite_replaces_existing_plain_lyrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            _add_plain_lyrics(path)
            write_sylt_tag(path, _SYNCED_LYRICS, embed_mode="overwrite")
            tags = ID3(path)

            uslt_frames = [v for k, v in tags.items() if k.startswith("USLT")]
            txxx_frames = [v for k, v in tags.items()
                           if k.startswith("TXXX") and v.desc.upper() == "LYRICS"]

            # The plain lyrics should have been replaced with LRC-timestamped text
            for frame in uslt_frames:
                assert _PLAIN_LYRICS not in frame.text, (
                    "Plain lyrics should be overwritten in USLT"
                )
            for frame in txxx_frames:
                text = frame.text[0] if frame.text else ""
                assert _PLAIN_LYRICS not in text, (
                    "Plain lyrics should be overwritten in TXXX:LYRICS"
                )

    def test_overwrite_is_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            # No embed_mode argument — should behave like "overwrite"
            write_sylt_tag(path, _SYNCED_LYRICS)
            tags = ID3(path)

            assert any(k.startswith("USLT") for k in tags), (
                "Default (overwrite) should write USLT"
            )
            assert any(
                k.startswith("TXXX") and tags[k].desc.upper() == "LYRICS"
                for k in tags
            ), "Default (overwrite) should write TXXX:LYRICS"


class TestWriteSyltTagSyltOnly:
    def test_sylt_only_writes_sylt_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            write_sylt_tag(path, _SYNCED_LYRICS, embed_mode="sylt_only")
            tags = ID3(path)

            sylt_keys = [k for k in tags if k.startswith("SYLT")]
            assert sylt_keys, "SYLT frame should be written"

    def test_sylt_only_does_not_write_uslt_or_txxx(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            write_sylt_tag(path, _SYNCED_LYRICS, embed_mode="sylt_only")
            tags = ID3(path)

            uslt_keys = [k for k in tags if k.startswith("USLT")]
            txxx_keys = [k for k in tags if k.startswith("TXXX:LYRICS")]

            assert not uslt_keys, "USLT should not be written in sylt_only mode"
            assert not txxx_keys, "TXXX:LYRICS should not be written in sylt_only mode"

    def test_sylt_only_preserves_existing_plain_lyrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            _add_plain_lyrics(path)
            write_sylt_tag(path, _SYNCED_LYRICS, embed_mode="sylt_only")
            tags = ID3(path)

            uslt_frames = [v for k, v in tags.items() if k.startswith("USLT")]
            txxx_frames = [v for k, v in tags.items()
                           if k.startswith("TXXX") and v.desc.upper() == "LYRICS"]

            assert uslt_frames, "Existing USLT should be preserved"
            assert any(_PLAIN_LYRICS in f.text for f in uslt_frames), (
                "Existing plain USLT lyrics should be unchanged"
            )
            assert txxx_frames, "Existing TXXX:LYRICS should be preserved"
            assert any(_PLAIN_LYRICS in (f.text[0] if f.text else "") for f in txxx_frames), (
                "Existing plain TXXX:LYRICS should be unchanged"
            )

    def test_sylt_only_replaces_existing_sylt_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_mp3(tmp)
            # Write an initial SYLT frame
            try:
                tags = ID3(path)
            except Exception:
                tags = ID3()
            tags.add(SYLT(
                encoding=Encoding.UTF8, lang="eng", desc="", format=2, type=1,
                text=[("Old line\n", 9999)],
            ))
            tags.save(path)

            write_sylt_tag(path, _SYNCED_LYRICS, embed_mode="sylt_only")
            tags = ID3(path)

            sylt_frames = [v for k, v in tags.items() if k.startswith("SYLT")]
            assert sylt_frames, "SYLT frame should exist after sylt_only write"
            # The old SYLT data should be replaced
            all_texts = [t for frame in sylt_frames for t, _ in frame.text]
            assert not any("Old line" in t for t in all_texts), (
                "Old SYLT frame should be replaced"
            )
