"""speech.py - speak text without freezing the video. Mac: `say`. Windows: pyttsx3."""
import subprocess
import sys
import threading


def speak(text):
    if sys.platform == "darwin":
        subprocess.Popen(["say", text])  # built-in macOS voice
        return

    def run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:  # speech must never crash the app
            print("Speech error:", exc)

    threading.Thread(target=run, daemon=True).start()
