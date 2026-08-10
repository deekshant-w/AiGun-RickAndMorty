import collections
import queue
import threading
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeModel
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import download_voice

from rnm.paths import TMP_DIR, TTS_MODEL_DIR

RATE = 16000  # 16kHz - default for openwakeword, webrtcvad, and whisper
BLOCK = 1280  # 80 ms of audio, what openwakeword likes to scan a rolling window finding the wake word
BLOCK_MS = BLOCK / RATE * 1000
MAX_LAG_BLOCKS = 15  # 15*(BLOCK/RATE) = 1.2s | how many blocks in the audio queue after which the system is concidered to have "lagged" and will reset the queue to avoid memory issues
VAD_FRAME = 320  # 20 ms, what webrtcvad accepts
WAKE_WORD = "hey_jarvis"  # also: alexa, hey_mycroft, hey_rhasspy
WAKE_THRESHOLD = 0.5
SILENCE_MS = 1000  # how long you stay quiet before it cuts the recording
MAX_UTTERANCE_S = 15  # hard stop, so it never records forever
NO_SPEECH_TIMEOUT_S = 3  # false wake, nobody said anything
PRE_ROLL_S = 0.2  # keep audio from before the trigger so words are not clipped
DISPLAY_AFTER_EVERY_N = 20  # Display user feedback after every N blocks of audio processed, so they know the system is still alive and listening

audio_q = queue.Queue()  # stores audio from the mic callback, so we can process it in the main thread


def defaultCallback(text: str):
    """Runs on a worker thread once you finish speaking."""
    print(f"[command] {text}")
    print("Waiting for 5 seconds...")

    low = text.lower()
    if "time" in low:
        print(time.strftime("It is %H:%M"))
    elif "stop" in low or "quit" in low or "exit" in low:
        exit(0)


def mic_callback(indata, frames, time_info, status):
    if status:
        print(status, flush=True)
    audio_q.put(bytes(indata))


def has_speech(vad: webrtcvad.Vad, block_bytes: bytes) -> bool:
    for i in range(0, len(block_bytes), VAD_FRAME * 2):
        frame = block_bytes[i : i + VAD_FRAME * 2]
        if len(frame) == VAD_FRAME * 2 and vad.is_speech(frame, RATE):
            return True
    return False


def transcribe_and_dispatch(stt: WhisperModel, pcm_bytes: bytes, on_command: Callable):
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = stt.transcribe(audio, language="en", beam_size=1)
    text = " ".join(s.text for s in segments).strip()
    on_command(text)


def drain_stale(q: queue.Queue, keep: int = 1) -> int:
    """Discard backlog, keeping the newest `keep` blocks. Returns count dropped."""
    dropped = 0
    while q.qsize() > keep:
        try:
            q.get_nowait()
        except queue.Empty:
            break
        dropped += 1
    return dropped


def inputLoop(callback: Callable):
    print("loading models...")
    wake = WakeModel(
        wakeword_models=[WAKE_WORD], inference_framework="onnx"
    )  # It has it's own internal buffer storing frames till it detects the wake word
    stt = WhisperModel("small.en", device="cpu", compute_type="int8")  # Audio to text
    vad = webrtcvad.Vad(
        2
    )  # Checks if a frame contains speech. 0 is least aggressive about filtering out non-speech, 3 is most aggressive.

    pre_roll = collections.deque(
        maxlen=max(1, round(PRE_ROLL_S / (BLOCK / RATE)))
    )  # store audio from before the wake word was detected, so we don't cut off the beginning of speech
    frames = []  # input for audio to text, stores audio after the wake word was detected
    wakeWordDetected = False  # True when wake word has been detected and we are recording audio for speech to text
    silence_ms = 0
    heard_voice = False  # True when speech is detected after wake word
    started = 0.0

    stream = sd.RawInputStream(
        samplerate=RATE,
        blocksize=BLOCK,
        channels=1,
        dtype="int16",
        callback=mic_callback,
    )

    with stream:
        print(f"listening for '{WAKE_WORD}'. ctrl+c to quit.")
        while True:
            if not wakeWordDetected and audio_q.qsize() > MAX_LAG_BLOCKS:
                dropped = drain_stale(audio_q)
                wake.reset()
                pre_roll.clear()
                frames = []
                print(f"System lagged, dropped {dropped} blocks...")

            block = audio_q.get()

            if not wakeWordDetected:
                pre_roll.append(block)
                samples = np.frombuffer(block, dtype=np.int16)
                score = max(wake.predict(samples).values())
                if score >= WAKE_THRESHOLD:  # Wake word detected
                    print(f"[wake] {score:.2f}")
                    wake.reset()
                    frames = list(pre_roll)
                    pre_roll.clear()
                    wakeWordDetected = True
                    heard_voice = False
                    silence_ms = 0
                    started = time.time()
                continue

            # Wake word has been detected, we are now recording audio for speech to text
            frames.append(block)
            elapsed = time.time() - started

            if has_speech(vad, block):
                heard_voice = True
                silence_ms = 0
            else:
                silence_ms += BLOCK_MS

            done = False
            if heard_voice and silence_ms >= SILENCE_MS:
                done = True
            elif not heard_voice and elapsed > NO_SPEECH_TIMEOUT_S:
                print("[wake] false alarm, back to listening")
                wakeWordDetected = False
                frames = []
                continue
            elif elapsed > MAX_UTTERANCE_S:
                done = True

            if done:
                frameInput = b"".join(frames)
                frames = []
                wakeWordDetected = False
                done = False
                transcribe_and_dispatch(stt, frameInput, callback)
                drain_stale(audio_q, 0)
            # else:
            #     if len(frames) > 0 and len(frames) % DISPLAY_AFTER_EVERY_N == 0:
            #         threading.Thread(
            #             target=transcribe_and_dispatch,
            #             args=(stt, b''.join(frames), lambda _: print(f">> [listening] {_}", flush=True)),
            #             daemon=True,
            #         ).start()


class PiperTTS:
    def __init__(self, voice_name: str = "en_US-hfc_female-medium"):
        self.voice_name = voice_name
        self.model_dir = TTS_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.syn_config = SynthesisConfig(length_scale=0.9, noise_scale=1.1, noise_w_scale=1.1, volume=0.4)

        model_path = self.model_dir / f"{voice_name}.onnx"
        config_path = self.model_dir / f"{voice_name}.onnx.json"

        if not model_path.exists() or not config_path.exists():
            download_voice(voice_name, self.model_dir)

        self.voice = PiperVoice.load(str(model_path))

    def play(self, text: str):
        stream = None
        for chunk in self.voice.synthesize(text, syn_config=self.syn_config):
            if stream is None:
                stream = sd.RawOutputStream(
                    samplerate=chunk.sample_rate,
                    channels=chunk.sample_channels,
                    dtype="int16",
                )
                stream.start()
            stream.write(chunk.audio_int16_bytes)
        if stream:
            stream.stop()
            stream.close()

    def play_async(self, text: str):
        import threading

        thread = threading.Thread(target=self.play, args=(text,))
        thread.start()


TTS = PiperTTS()


def laser(count=1, delay=0.2):
    # Play a laser sound effect `count` times.
    laser_path = TMP_DIR / "laser.raw"

    def _play():
        data = np.fromfile(laser_path, dtype=np.int16)
        with sd.OutputStream(samplerate=44100, channels=1, dtype="int16") as s:
            s.write(data)

    def _delayed_play():
        time.sleep(2)
        for _ in range(count):
            threading.Thread(target=_play).start()
            time.sleep(delay)

    threading.Thread(target=_delayed_play).start()


if __name__ == "__main__":
    # try:
    #     inputLoop(defaultCallback)
    # except KeyboardInterrupt:
    #     print("\nstopped")

    # tts = PiperTTS()
    # tts.play("Hello, I am your AI assistant. How can I help you today?")

    laser(5)
