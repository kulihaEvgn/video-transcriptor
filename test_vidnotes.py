#!/usr/bin/env python3
"""Проверки для мест, где ошибка не падает, а тихо портит результат.

Запуск: python test_vidnotes.py
"""
import tempfile
from pathlib import Path

from vidnotes import MOMENT, srt_time
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
