"""Offline voice commands with Vosk, from the phone or headset microphone.

Vosk is restricted to a small grammar (only the command phrases plus "[unk]" for
anything else). That makes recognition fast and accurate even with background
noise, and nothing leaves the laptop. The small English model (~40 MB) is
downloaded by Vosk on first use.

parse() maps recognised text to a Command and is unit-tested on its own.
"""

from __future__ import annotations

import json
import queue
import threading
import time

from src.commands import Command, CommandEvent

# Phrase -> command. Several phrasings per command make it natural to speak.
PHRASES: dict[str, Command] = {
    "take off": Command.TAKE_OFF,
    "follow me": Command.FOLLOW,
    "follow": Command.FOLLOW,
    "stop": Command.STOP,
    "hold": Command.STOP,
    "stay": Command.STOP,
    "land": Command.LAND,
    "land now": Command.LAND,
    "come back": Command.COME_BACK,
    "come here": Command.COME_BACK,
    "go higher": Command.HIGHER,
    "higher": Command.HIGHER,
    "go up": Command.HIGHER,
    "go lower": Command.LOWER,
    "lower": Command.LOWER,
    "go down": Command.LOWER,
    "closer": Command.CLOSER,
    "come closer": Command.CLOSER,
    "further": Command.FURTHER,
    "back off": Command.FURTHER,
    "take a photo": Command.PHOTO,
    "take photo": Command.PHOTO,
    "photo": Command.PHOTO,
}
SAMPLE_RATE = 16000


def parse(text: str) -> Command | None:
    """The command in a recognised phrase, preferring the longest matching phrase."""
    text = " ".join(text.lower().replace("[unk]", " ").split())
    if not text:
        return None
    for phrase in sorted(PHRASES, key=len, reverse=True):
        if f" {phrase} " in f" {text} ":
            return PHRASES[phrase]
    return None


class VoiceListener:
    """Background thread: microphone -> Vosk -> CommandEvents in `events`."""

    def __init__(self, device: int | None = None, min_confidence: float = 0.6):
        self.device = device
        self.min_confidence = min_confidence
        self.events: queue.Queue[CommandEvent] = queue.Queue()
        self.last_text = ""
        self.error: str | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            model = Model(lang="en-us")  # downloads the small English model on first use
            grammar = json.dumps(sorted(PHRASES) + ["[unk]"])
            recognizer = KaldiRecognizer(model, SAMPLE_RATE, grammar)
            recognizer.SetWords(True)
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=4000, dtype="int16", channels=1,
                                   device=self.device) as stream:
                while self._running:
                    data, _ = stream.read(4000)
                    if recognizer.AcceptWaveform(bytes(data)):
                        self._handle(json.loads(recognizer.Result()))
        except Exception as error:  # microphone missing, model download failed, ...
            self.error = f"{type(error).__name__}: {error}"

    def _handle(self, result: dict) -> None:
        text = result.get("text", "")
        if not text:
            return
        words = result.get("result", [])
        confidence = sum(w.get("conf", 0.0) for w in words) / len(words) if words else 0.0
        self.last_text = f"{text} ({confidence:.2f})"
        command = parse(text)
        if command is not None and confidence >= self.min_confidence:
            self.events.put(CommandEvent(command, "voice", time.monotonic()))
