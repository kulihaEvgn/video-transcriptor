#!/usr/bin/env python3
"""Окно для vidnotes: скачать модель → выбрать видео → сохранить результат."""
import multiprocessing
import queue
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from vidnotes import (
    DEFAULT_AGENT,
    MODEL_NAME,
    VIDEO_SUFFIXES,
    VidnotesError,
    download_model,
    human_size,
    model_is_downloaded,
    model_size,
    run,
)

LANGS = ["auto", "ru", "en", "de", "uk", "pl", "es", "fr"]
FILETYPES = [
    ("Видео", " ".join(f"*{s}" for s in VIDEO_SUFFIXES)),
    ("Все файлы", "*.*"),
]


class App:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.video = None
        self.result_dir = None
        self.workdir = None

        root.title("vidnotes")
        root.minsize(620, 480)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        # 1. модель
        self.model_label = ttk.Label(frame, text="")
        self.model_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.model_btn = ttk.Button(frame, text="Скачать модель", command=self.on_download)
        self.model_btn.grid(row=0, column=2, sticky="e", pady=(0, 6))

        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 12))

        # 2. видео и параметры
        self.video_label = ttk.Label(frame, text="Видео не выбрано")
        self.video_label.grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(frame, text="Выбрать видео…", command=self.on_pick_video).grid(
            row=2, column=2, sticky="e"
        )
        ttk.Label(frame, text=f"Читаем: {', '.join(s.lstrip('.') for s in VIDEO_SUFFIXES)}").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(2, 12)
        )

        params = ttk.Frame(frame)
        params.grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 12))
        ttk.Label(params, text="Язык речи:").pack(side="left")
        self.lang = tk.StringVar(value="ru")
        ttk.Combobox(params, textvariable=self.lang, values=LANGS, width=6, state="readonly").pack(
            side="left", padx=(4, 16)
        )
        ttk.Label(params, text="Сколько кадров брать — посчитается от длительности").pack(side="left")

        self.start_btn = ttk.Button(frame, text="Начать", command=self.on_start, state="disabled")
        self.start_btn.grid(row=5, column=0, columnspan=3, pady=(0, 12))

        # 3. лог и сохранение
        self.log_box = tk.Text(frame, height=14, wrap="word", state="disabled")
        self.log_box.grid(row=6, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(6, weight=1)
        scroll = ttk.Scrollbar(frame, command=self.log_box.yview)
        scroll.grid(row=6, column=3, sticky="ns")
        self.log_box["yscrollcommand"] = scroll.set

        self.save_btn = ttk.Button(
            frame, text="Сохранить результат…", command=self.on_save, state="disabled"
        )
        self.save_btn.grid(row=7, column=0, columnspan=3, pady=(12, 0))

        self.refresh_model_state()
        if not shutil.which(DEFAULT_AGENT.split()[0]):
            self.log(f"внимание: не нашёлся «{DEFAULT_AGENT.split()[0]}» — без него моменты выбрать нечем")
        self.root.after(100, self.drain)

    # --- события из рабочего потока ---

    def drain(self):
        while not self.events.empty():
            kind, payload = self.events.get()
            if kind == "log":
                self.log(payload)
            elif kind == "progress":
                self.progress["value"] = payload
            elif kind == "model_size":
                self.show_model_size(payload)
            elif kind == "model_done":
                self.refresh_model_state()
                self.busy(False)
                self.log("модель на месте")
            elif kind == "run_done":
                self.busy(False)
                self.save_btn["state"] = "normal"
                self.log("готово — жми «Сохранить результат…»")
            elif kind == "error":
                self.busy(False)
                self.log(f"ошибка: {payload}")
                messagebox.showerror("vidnotes", payload)
        self.root.after(100, self.drain)

    def log(self, text):
        self.log_box["state"] = "normal"
        self.log_box.insert("end", f"{text}\n")
        self.log_box.see("end")
        self.log_box["state"] = "disabled"

    def busy(self, on):
        state = "disabled" if on else "normal"
        self.start_btn["state"] = state if self.video else "disabled"
        self.model_btn["state"] = "disabled" if (on or model_is_downloaded()) else "normal"

    def work(self, fn):
        """Запускает долгую работу в фоне, чтобы окно не подвисало."""
        self.busy(True)

        def wrapped():
            try:
                fn()
            except VidnotesError as e:
                self.events.put(("error", str(e)))
            except Exception as e:  # noqa: BLE001 — в окне лучше показать, чем уронить
                self.events.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=wrapped, daemon=True).start()

    # --- кнопки ---

    def refresh_model_state(self):
        self.model_ready = model_is_downloaded()
        state = "скачана" if self.model_ready else "не скачана"
        self.model_label["text"] = f"Модель {MODEL_NAME}: {state}"
        self.model_btn["text"] = "Скачать модель"
        self.model_btn["state"] = "disabled" if self.model_ready else "normal"
        self.ask_model_size()

    def ask_model_size(self):
        """Вес модели узнаём в стороне: у скачанной он из кэша, у прочей — из сети."""

        def job():
            size = model_size()
            if size:
                self.events.put(("model_size", human_size(size)))

        threading.Thread(target=job, daemon=True).start()

    def show_model_size(self, size):
        if self.model_ready:
            self.model_label["text"] = f"Модель {MODEL_NAME}: скачана, занимает {size}"
        else:
            self.model_label["text"] = f"Модель {MODEL_NAME}: не скачана"
            self.model_btn["text"] = f"Скачать модель ({size})"

    def on_download(self):
        self.log(f"качаю модель {MODEL_NAME}…")
        self.progress["value"] = 0

        def job():
            download_model(MODEL_NAME, on_progress=lambda pct: self.events.put(("progress", pct)))
            self.events.put(("model_done", None))

        self.work(job)

    def on_pick_video(self):
        path = filedialog.askopenfilename(title="Выбери видео", filetypes=FILETYPES)
        if not path:
            return
        self.video = Path(path)
        self.video_label["text"] = f"Видео: {self.video.name}"
        self.save_btn["state"] = "disabled"
        self.result_dir = None
        self.busy(False)

    def on_start(self):
        if not model_is_downloaded():
            messagebox.showwarning("vidnotes", "Сначала скачай модель")
            return
        self.workdir = Path(tempfile.mkdtemp(prefix="vidnotes_"))
        lang, video = self.lang.get(), self.video

        def job():
            notes = run(
                video, lang, dest=self.workdir,
                log=lambda line: self.events.put(("log", line)),
            )
            self.result_dir = notes.parent
            self.events.put(("run_done", None))

        self.work(job)

    def on_save(self):
        target = filedialog.askdirectory(title="Куда сохранить результат")
        if not target or not self.result_dir:
            return
        moved, skipped = move_result(
            self.result_dir,
            Path(target),
            ask_overwrite=lambda name: messagebox.askyesno(
                "vidnotes", f"«{name}» уже есть. Перезаписать?"
            ),
        )
        self.log(f"сохранено в {target}" + (f", пропущено: {skipped}" if skipped else ""))
        self.save_btn["state"] = "disabled"
        if moved:
            reveal(Path(target))


def move_result(src_dir, target, ask_overwrite):
    """Переносит всё из рабочей папки в выбранную. Возвращает (перенесено, пропущено).

    Спрашивает перед перезаписью: результат чужого прогона молча затирать нельзя.
    """
    target.mkdir(parents=True, exist_ok=True)
    moved, skipped = [], []
    for item in sorted(src_dir.iterdir()):
        dst = target / item.name
        if dst.exists():
            if not ask_overwrite(item.name):
                skipped.append(item.name)
                continue
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.move(str(item), str(dst))
        moved.append(dst)
    return moved, skipped


def reveal(path):
    """Открыть папку в проводнике — на всех трёх системах по-своему."""
    import sys

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        elif sys.platform.startswith("win"):
            subprocess.run(["explorer", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])
    except OSError:
        pass


def main():
    multiprocessing.freeze_support()  # иначе в сборке дочерний процесс открывает второе окно
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
