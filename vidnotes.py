#!/usr/bin/env python3
"""vidnotes — транскрипт видео + скриншоты ключевых мест.

Ядро и командная строка. Окно с кнопками — в vidnotes_gui.py.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MOMENT = re.compile(r"^(\d{2}:\d{2}:\d{2})(?:[.,]\d{1,3})?\s*\|\s*(.+)$")
DEFAULT_AGENT = "claude -p"
MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv")

PROMPT = """Below is an SRT transcript of a video. Pick the {count} most important moments \
— where something is decided, shown, demonstrated, or the topic shifts. Spread them across \
the whole video, not just the start.
Output ONLY lines in the exact format:
HH:MM:SS | short caption in the transcript's own language
No preamble, no numbering, no other text."""


def auto_count(duration_seconds):
    """Сколько моментов брать: примерно один на две минуты записи.

    Пользователь не знает заранее, сколько в видео важного, — знает только длительность.
    """
    return max(5, min(30, round(duration_seconds / 120)))


class VidnotesError(Exception):
    """Понятная пользователю ошибка: показывается и в консоли, и в окне."""


def srt_time(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# что именно качать: имена вроде large-v3-turbo живут в разных репозиториях,
# поэтому маппинг и список файлов берём у самой faster-whisper, а не выдумываем
MODEL_FILES = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def _repo_id(name):
    try:
        from faster_whisper.utils import _MODELS

        return _MODELS.get(name, name)
    except ImportError:
        return name


def model_is_downloaded(name=MODEL_NAME):
    """Лежит ли модель в кэше — чтобы не лезть в сеть без спроса."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(_repo_id(name), allow_patterns=MODEL_FILES, local_files_only=True)
        return True
    except Exception:
        return False


def download_model(name=MODEL_NAME, tqdm_class=None):
    from huggingface_hub import snapshot_download

    kwargs = {"tqdm_class": tqdm_class} if tqdm_class else {}
    snapshot_download(_repo_id(name), allow_patterns=MODEL_FILES, **kwargs)


def transcribe(video, lang, out, log=print, model_name=MODEL_NAME):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="auto", compute_type="auto")
    segments, info = model.transcribe(
        str(video), language=None if lang == "auto" else lang, beam_size=5
    )
    log(f"модель {model_name}, язык {info.language}, длительность {info.duration:.0f} с")
    duration = info.duration

    srt, txt = [], []
    for i, seg in enumerate(segments, 1):
        text = seg.text.strip()
        log(f"[{srt_time(seg.start)}] {text}")
        srt.append(f"{i}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n{text}\n")
        txt.append(text)
    if not srt:
        raise VidnotesError("речи в файле не нашлось")
    (out / "transcript.srt").write_text("\n".join(srt), encoding="utf-8")
    (out / "transcript.txt").write_text("\n".join(txt), encoding="utf-8")
    return duration


def pick_moments(out, count, agent=DEFAULT_AGENT, log=print):
    """Отдаёт агенту транскрипт, забирает строки «таймкод | подпись»."""
    prompt = PROMPT.format(count=count)
    srt = (out / "transcript.srt").read_text(encoding="utf-8")

    if agent == DEFAULT_AGENT:
        # родной путь: промпт аргументом, транскрипт на stdin, ответ с замером токенов
        raw = _run_agent(agent.split() + ["--output-format", "json", prompt], srt)
        (out / "agent.json").write_text(raw, encoding="utf-8")
        try:
            data = json.loads(raw)
            answer = data.get("result", "")
            u = data["usage"]
            spent = u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"]
            log(
                f"{spent} токенов на вход ({u['cache_read_input_tokens']} из кэша), "
                f"{u['output_tokens']} на выход, ${data['total_cost_usd']}"
            )
        except (json.JSONDecodeError, KeyError):
            answer = raw
    else:
        # любой другой агент: всё одним куском на stdin, ответ текстом на stdout
        answer = _run_agent(agent.split(), f"{prompt}\n\n{srt}")

    (out / "agent.txt").write_text(answer, encoding="utf-8")
    moments = [m.groups() for m in map(MOMENT.match, answer.splitlines()) if m][:count]
    if not moments:
        raise VidnotesError(f"агент не вернул таймкоды — смотри {out / 'agent.txt'}")
    (out / "moments.txt").write_text(
        "".join(f"{ts} | {caption}\n" for ts, caption in moments), encoding="utf-8"
    )
    return moments


def _run_agent(cmd, stdin_text):
    try:
        done = subprocess.run(
            cmd, input=stdin_text, capture_output=True, text=True, encoding="utf-8"
        )
    except FileNotFoundError:
        raise VidnotesError(f"не нашёлся агент «{cmd[0]}» — поставь его или задай VIDNOTES_AGENT")
    if done.returncode != 0:
        raise VidnotesError(f"агент «{cmd[0]}» вернул ошибку: {done.stderr.strip()[:300]}")
    return done.stdout


def ts_seconds(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def save_jpeg(frame, path):
    """Кодируем кадр тем же av, что уже декодирует видео — второй ffmpeg в сборке не нужен."""
    import av

    with av.open(str(path), "w", format="mjpeg") as out:
        stream = out.add_stream("mjpeg", rate=1)
        stream.width, stream.height = frame.width, frame.height
        stream.pix_fmt = "yuvj420p"
        stream.codec_context.qmin = stream.codec_context.qmax = 2  # как -q:v 2 у ffmpeg
        frame = frame.reformat(format="yuvj420p")
        frame.pts = None
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def grab_frames(video, out, moments, log=print):
    import av

    shots = out / "shots"
    shots.mkdir(exist_ok=True)
    names = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for i, (ts, _) in enumerate(moments, 1):
            want = ts_seconds(ts)
            # seek встаёт на ключевой кадр не позже нужной секунды, дальше доходим декодированием
            container.seek(int(want / stream.time_base), stream=stream)
            frame = last = None
            for candidate in container.decode(stream):
                last = candidate
                if candidate.time is not None and candidate.time >= want - 0.001:
                    frame = candidate
                    break
            frame = frame or last
            if frame is None:
                log(f"кадр на {ts} не нашёлся, пропускаю")
                continue
            name = f"{i:02d}_{ts.replace(':', '-')}.jpg"
            save_jpeg(frame, shots / name)
            names.append(name)
    log(f"кадров вырезано: {len(names)}")
    return names


def run(video, lang="auto", count=None, agent=DEFAULT_AGENT, dest=None, log=print):
    """Весь путь от видео до markdown. Возвращает путь к .md."""
    video = Path(video)
    if not video.is_file():
        raise VidnotesError(f"нет файла: {video}")
    base = Path(dest) if dest else video.parent
    out = base / (video.stem + "_notes")
    out.mkdir(parents=True, exist_ok=True)

    log("[1/3] транскрипт…")
    duration = transcribe(video, lang, out, log=log)

    if count is None:
        count = auto_count(duration)
        log(f"беру {count} моментов — по одному примерно на две минуты записи")
    log(f"[2/3] {agent} выбирает моменты…")
    moments = pick_moments(out, count, agent, log=log)

    log("[3/3] скриншоты…")
    names = grab_frames(video, out, moments, log=log)
    moments = moments[: len(names)]

    lines = [f"# {video.name}", ""]
    for (ts, caption), name in zip(moments, names):
        lines += [f"## {ts} — {caption}", "", f"![{ts} — {caption}]({out.name}/shots/{name})", ""]
    lines += [
        "---", "", "## Полный транскрипт", "",
        (out / "transcript.txt").read_text(encoding="utf-8"),
    ]

    notes = base / (video.stem + "_notes.md")
    notes.write_text("\n".join(lines), encoding="utf-8")
    log(f"готово: {notes}")
    return notes


def main():
    ap = argparse.ArgumentParser(
        prog="vidnotes", description="Видео → markdown: транскрипт и скриншоты ключевых мест"
    )
    ap.add_argument("video", type=Path)
    ap.add_argument("lang", nargs="?", default="auto", help="язык речи: ru, en, … (по умолчанию auto)")
    ap.add_argument(
        "count", nargs="?", type=int,
        help="сколько моментов (по умолчанию — по одному на две минуты записи)",
    )
    ap.add_argument("-o", "--out", type=Path, help="куда положить результат (по умолчанию рядом с видео)")
    args = ap.parse_args()
    try:
        run(
            args.video, args.lang, args.count,
            agent=os.environ.get("VIDNOTES_AGENT", DEFAULT_AGENT),
            dest=args.out,
        )
    except VidnotesError as e:
        sys.exit(f"vidnotes: {e}")


if __name__ == "__main__":
    main()
