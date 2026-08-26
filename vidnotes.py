#!/usr/bin/env python3
"""vidnotes — транскрипт видео + скриншоты ключевых мест.

Ядро и командная строка. Окно с кнопками — в vidnotes_gui.py.
"""
import argparse
import json
import multiprocessing
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


MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo-q5_0")


def model_path(name=MODEL_NAME):
    """Путь к файлу модели. В WHISPER_MODEL можно положить и готовый путь к .bin."""
    from pywhispercpp.constants import MODELS_DIR

    direct = Path(name).expanduser()
    if direct.is_file():
        return direct
    return Path(MODELS_DIR) / f"ggml-{name}.bin"


def model_url(name=MODEL_NAME):
    from pywhispercpp.constants import MODELS_BASE_URL, MODELS_PREFIX_URL

    return f"{MODELS_BASE_URL}/{MODELS_PREFIX_URL}-{name}.bin"


def model_is_downloaded(name=MODEL_NAME):
    return model_path(name).is_file()


def model_size(name=MODEL_NAME):
    """Вес модели: с диска, если она уже есть, иначе спрашиваем у сервера.

    Отдаёт None, если файла нет и до сети не достучались — врать оценкой не нужно.
    """
    path = model_path(name)
    if path.is_file():
        return path.stat().st_size
    try:
        import urllib.request

        request = urllib.request.Request(model_url(name), method="HEAD")
        with urllib.request.urlopen(request, timeout=15) as response:
            return int(response.headers.get("Content-Length") or 0) or None
    except Exception:
        return None


def download_model(name=MODEL_NAME, on_progress=None):
    """Качаем в .part и переименовываем в конце: оборванная закачка не притворится моделью."""
    import urllib.request

    path = model_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    with urllib.request.urlopen(model_url(name), timeout=60) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if on_progress and total:
                on_progress(done / total * 100)
    part.rename(path)


def human_size(num_bytes):
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024 ** 3:.1f} ГБ".replace(".", ",")
    return f"{num_bytes / 1024 ** 2:.0f} МБ"


def video_duration(video):
    import av

    with av.open(str(video)) as container:
        return (container.duration or 0) / 1_000_000


def transcribe(video, lang, out, log=print, model_name=MODEL_NAME):
    from pywhispercpp.model import Model

    if not model_is_downloaded(model_name):
        raise VidnotesError(f"модель не скачана: {model_path(model_name)}")

    model = Model(str(model_path(model_name)), language=None if lang == "auto" else lang,
                  print_progress=False, redirect_whispercpp_logs_to=False)
    log(f"модель {model_name}, язык {lang}")

    srt, txt = [], []

    def on_segment(segment):
        text = segment.text.strip()
        # таймкоды whisper.cpp приходят в сотых долях секунды
        start, finish = segment.t0 / 100, segment.t1 / 100
        log(f"[{srt_time(start)}] {text}")
        srt.append(f"{len(srt) + 1}\n{srt_time(start)} --> {srt_time(finish)}\n{text}\n")
        txt.append(text)

    model.transcribe(str(video), new_segment_callback=on_segment)
    if not srt:
        raise VidnotesError("речи в файле не нашлось")
    (out / "transcript.srt").write_text("\n".join(srt), encoding="utf-8")
    (out / "transcript.txt").write_text("\n".join(txt), encoding="utf-8")
    return video_duration(video)


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


def prepare(video, lang="auto", dest=None, log=print):
    """Первая половина: транскрипт. Отдаёт папку, длительность и сколько моментов брать."""
    video = Path(video)
    if not video.is_file():
        raise VidnotesError(f"нет файла: {video}")
    base = Path(dest) if dest else video.parent
    out = base / (video.stem + "_notes")
    out.mkdir(parents=True, exist_ok=True)

    log("[1/3] транскрипт…")
    duration = transcribe(video, lang, out, log=log)
    return out, auto_count(duration)


def assemble(video, out, moments, log=print):
    """Вторая половина: кадры и markdown по готовому списку моментов."""
    video = Path(video)
    names = grab_frames(video, out, moments, log=log)
    moments = moments[: len(names)]

    lines = [f"# {video.name}", ""]
    for (ts, caption), name in zip(moments, names):
        lines += [f"## {ts} — {caption}", "", f"![{ts} — {caption}]({out.name}/shots/{name})", ""]
    lines += [
        "---", "", "## Полный транскрипт", "",
        (out / "transcript.txt").read_text(encoding="utf-8"),
    ]

    notes = out.parent / (video.stem + "_notes.md")
    notes.write_text("\n".join(lines), encoding="utf-8")
    log(f"готово: {notes}")
    return notes


def run(video, lang="auto", count=None, agent=DEFAULT_AGENT, dest=None, log=print):
    """Весь путь целиком, с выбором моментов через внешнего агента."""
    out, auto = prepare(video, lang, dest, log=log)
    if count is None:
        count = auto
        log(f"беру {count} моментов — по одному примерно на две минуты записи")
    log(f"[2/3] {agent} выбирает моменты…")
    moments = pick_moments(out, count, agent, log=log)
    return assemble(video, out, moments, log=log)


def read_moments(path):
    """Читает «HH:MM:SS | подпись» из файла — им плагин передаёт выбор Claude."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    moments = [m.groups() for m in map(MOMENT.match, lines) if m]
    if not moments:
        raise VidnotesError(f"в {path} нет строк вида «HH:MM:SS | подпись»")
    return moments


def main():
    # в собранном виде дочерний процесс перезапускает сам бинарь: без этого он
    # прилетает в argparse с флагами интерпретатора и роняет разбор аргументов
    multiprocessing.freeze_support()

    ap = argparse.ArgumentParser(
        prog="vidnotes", description="Видео → markdown: транскрипт и скриншоты ключевых мест"
    )
    ap.add_argument("video", type=Path, nargs="?")
    ap.add_argument("lang", nargs="?", default="auto", help="язык речи: ru, en, … (по умолчанию auto)")
    ap.add_argument(
        "count", nargs="?", type=int,
        help="сколько моментов (по умолчанию — по одному на две минуты записи)",
    )
    ap.add_argument("-o", "--out", type=Path, help="куда положить результат (по умолчанию рядом с видео)")
    ap.add_argument(
        "--transcribe-only", action="store_true",
        help="только расшифровать: моменты выберет тот, кто вызвал (так работает плагин)",
    )
    ap.add_argument(
        "--moments", type=Path,
        help="взять моменты из файла «HH:MM:SS | подпись» и собрать markdown",
    )
    ap.add_argument(
        "--download-model", action="store_true", help="скачать модель и выйти"
    )
    args = ap.parse_args()

    try:
        if args.download_model:
            if model_is_downloaded():
                print(f"модель уже на месте: {model_path()}")
                return
            size = model_size()
            print(f"качаю {MODEL_NAME}" + (f" ({human_size(size)})" if size else "") + "…")
            download_model(on_progress=lambda pct: print(f"\r{pct:.0f}%", end="", flush=True))
            print(f"\rготово: {model_path()}")
            return

        if not args.video:
            ap.error("нужен файл видео")

        if args.transcribe_only:
            out, count = prepare(args.video, args.lang, args.out)
            print(f"транскрипт: {out / 'transcript.srt'}")
            print(f"моментов брать: {args.count or count}")
        elif args.moments:
            out = (args.out or args.video.parent) / (args.video.stem + "_notes")
            assemble(args.video, out, read_moments(args.moments))
        else:
            run(
                args.video, args.lang, args.count,
                agent=os.environ.get("VIDNOTES_AGENT", DEFAULT_AGENT),
                dest=args.out,
            )
    except VidnotesError as e:
        sys.exit(f"vidnotes: {e}")


if __name__ == "__main__":
    main()
