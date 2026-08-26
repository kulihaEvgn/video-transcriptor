#!/usr/bin/env python3
"""Проверки для мест, где ошибка не падает, а тихо портит результат.

Запуск: python test_vidnotes.py
"""
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import vidnotes
from vidnotes import (
    AUDIO_RATE,
    CLAUDE_AGENT,
    MOMENT,
    VidnotesError,
    assemble,
    auto_count,
    pick_moments,
    read_moments,
    srt_time,
    transcribe,
    ts_seconds,
)
from vidnotes_gui import move_result


def test_moment_parsing():
    # агент повторяет формат SRT — с миллисекундами через запятую
    assert MOMENT.match("00:08:12,260 | Решили убрать лимит").groups() == (
        "00:08:12",
        "Решили убрать лимит",
    )
    assert MOMENT.match("01:02:03 | Подпись").groups() == ("01:02:03", "Подпись")
    # преамбула и нумерация в ответе не должны попадать в моменты
    assert MOMENT.match("Вот важные моменты:") is None
    assert MOMENT.match("1. 00:00:10 | Подпись") is None


def test_srt_time():
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(3661.5) == "01:01:01,500"


def test_ts_seconds():
    assert ts_seconds("00:00:00") == 0
    assert ts_seconds("01:02:03") == 3723
    assert ts_seconds("00:00:08.26") == 8.26
    assert ts_seconds("00:00:08,260") == 8.26  # запятая — как в SRT


def test_auto_count():
    # один момент на две минуты, но не меньше пяти и не больше тридцати
    assert auto_count(0) == 5
    assert auto_count(10 * 60) == 5
    assert auto_count(20 * 60) == 10
    assert auto_count(10 * 3600) == 30


def test_pick_moments_trims_to_count():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "transcript.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nречь\n", encoding="utf-8")
        # поддельный агент: болтает лишнее и присылает четыре момента вместо двух
        fake = out / "agent.py"
        fake.write_text(
            "import sys; sys.stdin.read()\n"
            "print('Вот важные моменты:')\n"
            "print('00:00:01,500 | Раз')\n"
            "print('00:00:02 | Два')\n"
            "print('00:00:03 | Три')\n",
            encoding="utf-8",
        )
        moments = pick_moments(out, 2, agent=f"{sys.executable} {fake}", log=lambda _: None)
        assert moments == [("00:00:01", "Раз"), ("00:00:02", "Два")]
        assert (out / "moments.txt").read_text(encoding="utf-8") == "00:00:01 | Раз\n00:00:02 | Два\n"


def test_pick_moments_without_timecodes_fails():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "transcript.srt").write_text("речь", encoding="utf-8")
        fake = out / "agent.py"
        fake.write_text("import sys; sys.stdin.read(); print('не понял задачу')\n", encoding="utf-8")
        try:
            pick_moments(out, 3, agent=f"{sys.executable} {fake}", log=lambda _: None)
        except VidnotesError:
            pass
        else:
            raise AssertionError("молчаливый агент должен ронять прогон с понятной ошибкой")


def test_read_moments():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "moments.txt"
        path.write_text("# заголовок\n00:01:02 | Подпись\n\n", encoding="utf-8")
        assert read_moments(path) == [("00:01:02", "Подпись")]
        path.write_text("тут нет таймкодов\n", encoding="utf-8")
        try:
            read_moments(path)
        except VidnotesError:
            pass
        else:
            raise AssertionError("файл без таймкодов должен ронять сборку")


def test_assemble_keeps_captions_with_their_frames():
    """Пропущенный кадр в середине не должен сдвигать подписи у следующих."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "video_notes"
        (out / "shots").mkdir(parents=True)
        (out / "transcript.txt").write_text("речь", encoding="utf-8")
        moments = [("00:00:01", "Раз"), ("00:00:02", "Два"), ("00:00:03", "Три [спорно]")]
        real = vidnotes.grab_frames
        # второй кадр не нашёлся — так grab_frames и ведёт себя на битом видео
        vidnotes.grab_frames = lambda video, out, moments, log=print: [
            (moments[0], "01_00-00-01.jpg"),
            (moments[2], "03_00-00-03.jpg"),
        ]
        try:
            notes = assemble(Path(tmp) / "video.mp4", out, moments, log=lambda _: None)
        finally:
            vidnotes.grab_frames = real
        text = notes.read_text(encoding="utf-8")
        assert "## 00:00:03 — Три [спорно]" in text
        assert "Два" not in text
        # скобки из подписи в ссылку не попадают, иначе картинка не отрисуется
        assert "![00:00:03](video_notes/shots/03_00-00-03.jpg)" in text


def test_assemble_without_transcript_fails():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            assemble(Path(tmp) / "video.mp4", Path(tmp) / "video_notes", [("00:00:01", "Раз")])
        except VidnotesError:
            pass
        else:
            raise AssertionError("сборка без транскрипта должна давать понятную ошибку")


def test_pick_moments_survives_unknown_usage_shape():
    """Замер токенов может съехать в новой версии CLI — моменты из-за этого терять нельзя."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "transcript.srt").write_text("речь", encoding="utf-8")
        real = vidnotes._run_agent
        vidnotes._run_agent = lambda cmd, stdin: json.dumps({"result": "00:00:05 | Показал деплой"})
        try:
            assert pick_moments(out, 3, agent=CLAUDE_AGENT, log=lambda _: None) == [
                ("00:00:05", "Показал деплой")
            ]
        finally:
            vidnotes._run_agent = real


def test_transcribe_feeds_samples_not_a_path():
    """whisper.cpp получает массив, а не путь: с путём pywhispercpp зовёт системный
    ffmpeg, которого у человека, поставившего плагин, обычно нет."""
    seen = {}

    class FakeModel:
        def __init__(self, *a, **kw):
            pass

        def transcribe(self, media, new_segment_callback=None, **kw):
            seen["media"] = media
            new_segment_callback(SimpleNamespace(text=" привет ", t0=0, t1=220))

    pkg, mod = types.ModuleType("pywhispercpp"), types.ModuleType("pywhispercpp.model")
    mod.Model, pkg.model = FakeModel, mod
    saved = {k: sys.modules.get(k) for k in ("pywhispercpp", "pywhispercpp.model")}
    sys.modules.update({"pywhispercpp": pkg, "pywhispercpp.model": mod})
    real = (vidnotes.model_is_downloaded, vidnotes.model_path, vidnotes.read_audio)
    vidnotes.model_is_downloaded = lambda *a, **kw: True
    vidnotes.model_path = lambda *a, **kw: Path("модель.bin")
    vidnotes.read_audio = lambda *a, **kw: [0.0] * (AUDIO_RATE * 90)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            duration = transcribe(Path("видео.mp4"), "ru", out, log=lambda _: None)
            assert not isinstance(seen["media"], (str, Path)), "путь вернёт зависимость от ffmpeg"
            assert duration == 90  # длительность считаем по звуку, а не по метаданным контейнера
            assert (out / "transcript.srt").read_text(encoding="utf-8") == (
                "1\n00:00:00,000 --> 00:00:02,200\nпривет\n"
            )
            assert (out / "transcript.txt").read_text(encoding="utf-8") == "привет"
    finally:
        vidnotes.model_is_downloaded, vidnotes.model_path, vidnotes.read_audio = real
        for name, module in saved.items():
            sys.modules.pop(name, None) if module is None else sys.modules.update({name: module})


def test_move_result_keeps_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, dst = tmp / "work", tmp / "out"
        (src / "shots").mkdir(parents=True)
        (src / "notes.md").write_text("новое", encoding="utf-8")
        (src / "shots" / "01.jpg").write_bytes(b"jpg")
        dst.mkdir()
        (dst / "notes.md").write_text("старое", encoding="utf-8")

        # отказ от перезаписи оставляет старый файл на месте, остальное переносит
        moved, skipped = move_result(src, dst, ask_overwrite=lambda name: False)
        assert skipped == ["notes.md"]
        assert (dst / "notes.md").read_text(encoding="utf-8") == "старое"
        assert (dst / "shots" / "01.jpg").exists()
        assert len(moved) == 1

        # согласие — перезаписывает
        (src / "notes.md").write_text("новое", encoding="utf-8")
        moved, skipped = move_result(src, dst, ask_overwrite=lambda name: True)
        assert not skipped
        assert (dst / "notes.md").read_text(encoding="utf-8") == "новое"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("всё сошлось")
