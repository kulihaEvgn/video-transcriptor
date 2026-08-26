# vidnotes

Video → markdown: a transcript plus screenshots of the moments that matter. Speech is recognized locally, and which moments matter is decided by Claude from the transcript text.

Installs as a Claude Code plugin — nothing to build, sign, or allow through your OS.

## Install

```
/plugin marketplace add kulihaEvgn/video-transcriptor
/plugin install vidnotes@vidnotes
```

You also need [uv](https://docs.astral.sh/uv/) — it brings up the environment with the recognizer on the fly: `brew install uv` on macOS, `winget install astral-sh.uv` on Windows. One time only.

The model (about 550 MB) is downloaded on the first video, after asking. It stays on disk after that.

## Using it

Just tell Claude what you want:

```
make notes from the recording ~/Downloads/standup.mp4, it's in Russian
what did we discuss in the meeting? the recording is on my desktop
go through ~/Movies/demo.mov, it's in English
```

Claude transcribes the recording, picks the key moments from the text itself, cuts the frames and hands back markdown: a heading per moment, the screenshot under it, the full transcript at the end.

Recognition runs locally; the only thing that leaves your machine is the transcript text — and it goes to the same Claude already working in your terminal.

Log lines, error messages and the desktop window are in Russian: the tool was written for a Russian-speaking team. Captions follow the language of the recording.

## What you get

Next to the video:

```
standup.mp4
standup_notes.md          ← this is what you read
standup_notes/
  transcript.srt          lines with timecodes
  transcript.txt          plain text
  moments.txt             the picked moments: timecode | caption
  shots/01_00-12-30.jpg   frames: order number and the moment's timecode
  …
```

Images are linked by relative path, so the `_notes` folder has to travel together with the `.md`.

## How long it takes

Measured on an M1 Pro: a 43-minute screen recording, 2.5 GB, audio over a video call.

| Step | Time |
| --- | --- |
| transcription | 3 min 25 s (12× faster than real time) |
| cutting frames | under a second |
| output folder | 12 MB for 30 frames |

On a Mac the GPU does the work (Metal); on Windows and Linux it is the CPU, and it takes several times longer. If you don't want to wait, take a smaller model: `WHISPER_MODEL=small` or `medium` — faster, and noticeably worse on Russian.

## Without Claude Code

The plugin isn't the only way in — inside it is a plain Python script.

**Command line.** Here the moments are picked not by your Claude but by a separately launched `claude -p`, and that costs roughly fifty cents per hour of recording, because the CLI sends its own system prompt on every call.

```bash
uv tool install git+https://github.com/kulihaEvgn/video-transcriptor
vidnotes standup.mp4 ru          # the number of moments follows the duration
vidnotes demo.mov en 6 -o ~/Desktop
```

**A window with buttons:** `vidnotes-gui` — download the model, pick a video, save the result. There are no prebuilt apps: an unsigned one is blocked by both Windows and macOS, and telling people to switch protection off for this is not okay.

### Environment variables

`WHISPER_MODEL` — which model to use. A name (`small`, `medium`, `large-v3-turbo-q5_0`) or a path to a `.bin` file you already have.

`VIDNOTES_AGENT` — what picks the moments where Claude Code isn't involved: the command line and the window. `claude -p` by default. Any other command gets the prompt with the transcript on stdin and must return `HH:MM:SS | caption` lines:

```bash
VIDNOTES_AGENT="ollama run llama3" vidnotes standup.mp4 ru
```

## Troubleshooting

**The transcript came out in the wrong language** — name the language explicitly; on `auto` whisper sometimes gets the first seconds wrong and ruins the rest.

**Frames are empty or black** — the moment landed on a fade between scenes. Take the timecode from `moments.txt` and cut the frame by hand: `ffmpeg -ss 00:12:30 -i standup.mp4 -frames:v 1 frame.jpg`.

**Transcription takes forever** — that's a CPU with no accelerator; take a smaller model via `WHISPER_MODEL`.

**It stops and says the model isn't downloaded** — run the analysis again and allow the download, or fetch it by hand: `vidnotes --download-model`.

## What it doesn't do

No speaker diarization, no splitting of long recordings into chunks (the transcript goes in whole — three hours of speech still fits), and no dropping of near-duplicate frames.
