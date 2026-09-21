"""Tests ensuring subprocess and file reading survive non-UTF-8 bytes (Latin-1 / corrupted ID3 tags)."""

import os
import subprocess
import tempfile
from pathlib import Path

import mutagen.id3 as id3

from app.alignment import _convert_to_wav, _get_audio_duration_ms


def test_convert_to_wav_with_non_utf8_metadata(tmp_path):
    mp3_path = str(tmp_path / "corrupt_tag.mp3")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:a",
            "libmp3lame",
            mp3_path,
        ],
        check=True,
        capture_output=True,
    )

    # Inject ISO-8859-1 byte 0xc3 (Ã) followed by ASCII space (invalid UTF-8 continuation)
    tags = id3.ID3(mp3_path)
    tags.add(id3.TIT2(encoding=0, text=["Track \xc3 Title"]))
    tags.save()

    wav_path = _convert_to_wav(mp3_path, str(tmp_path), "test-unicode")
    assert os.path.exists(wav_path)

    dur = _get_audio_duration_ms(wav_path)
    assert dur > 0


def test_lyrics_read_text_resilience(tmp_path):
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_bytes(b"Line 1 with \xc3 byte\nLine 2")
    text = lyrics_file.read_text(encoding="utf-8", errors="replace").strip()
    assert "\ufffd" in text
    assert "Line 2" in text
