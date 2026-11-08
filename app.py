"""
app.py - SVS Symposium 2026: ASL letters + gesture phrases + spelling game.

Run: python app.py

Keys:  1 = Letters   2 = Phrases   3 = Game
       m = mute      c = clear text / phrase history
       space = add a space (Letters)   s = speak the typed text (Letters)
       r = reset (Letters: clear text, Game: score to 0)   n = next word (Game)
       f = full screen on/off          q = quit
"""
import subprocess
import sys
import time

import cv2
import mediapipe as mp

from game import WordGame
from letters import LetterRecognizer
from phrases import PhraseRecognizer
from speech import speak

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE = "SVS Symposium 2026 - ASL & Gesture Communicator"
MODES = {1: "Letters", 2: "Phrases", 3: "Game"}
GREEN, YELLOW, WHITE, GREY, RED = (0, 220, 0), (0, 220, 255), (255, 255, 255), (200, 200, 200), (0, 0, 255)

# All positions and sizes below are written for a 480-pixel-high picture and are
# scaled automatically to the real window size.
H = 480


def unit(frame):
    return frame.shape[0] / H


def text_w(frame, text, scale=0.8, thick=2):
    u = unit(frame)
    return cv2.getTextSize(text, FONT, scale * u, max(1, round(thick * u)))[0][0] / u


def put(frame, text, org, scale=0.8, color=WHITE, thick=2):
    u = unit(frame)
    org = (int(org[0] * u), int(org[1] * u))
    cv2.putText(frame, text, org, FONT, scale * u, (0, 0, 0), max(1, round((thick + 3) * u)), cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale * u, color, max(1, round(thick * u)), cv2.LINE_AA)


def put_centered(frame, text, y, scale=0.8, color=WHITE, thick=2):
    dw = frame.shape[1] / unit(frame)
    put(frame, text, ((dw - text_w(frame, text, scale, thick)) / 2, y), scale, color, thick)


def draw_mode_bar(frame, mode):
    x = 10
    for key, name in MODES.items():
        label = f"[{key}] {name}"
        put(frame, label, (x, 28), 0.6, YELLOW if key == mode else GREY, 2)
        x += text_w(frame, label, 0.6, 2) + 20


def draw_word(frame, word, pos, y):
    scale, thick = 2.4, 4
    dw = frame.shape[1] / unit(frame)
    widths = [text_w(frame, ch, scale, thick) + 18 for ch in word]
    x = (dw - sum(widths)) / 2
    u = unit(frame)
    for i, (ch, w) in enumerate(zip(word, widths)):
        color = GREEN if i < pos else (YELLOW if i == pos else WHITE)
        put(frame, ch, (x, y), scale, color, thick)
        if i == pos:
            cv2.line(frame, (int(x * u), int((y + 12) * u)), (int((x + w - 18) * u), int((y + 12) * u)),
                     YELLOW, max(1, round(4 * u)))
        x += w


def screen_size():
    """Screen size in pixels. Asked in a separate process so Tk cannot clash with OpenCV."""
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import tkinter as t; r=t.Tk(); r.withdraw(); print(r.winfo_screenwidth(), r.winfo_screenheight())"],
            capture_output=True, text=True, timeout=10).stdout.split()
        w, h = int(out[0]), int(out[1])
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1280, 720  # unknown screen: assume a modest one


def fit_window(aspect, screen=None):
    """Largest window with the camera's aspect ratio that fits the screen (with room for
    the title bar, menu bar and dock). Returns (width, height) in pixels."""
    sw, sh = screen or screen_size()
    w = min(sw * 0.92, sh * 0.80 * aspect)
    w = max(640, int(w))
    return w, int(w / aspect)


def main():
    print("Loading letter model (first run downloads it)...")
    letters = LetterRecognizer()
    phrases = PhraseRecognizer()
    game = WordGame()
    if not phrases.available:
        print("No phrase model yet (model.joblib). Letters and Game still work.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Could not open the camera. Check camera permissions.")
        return

    display = None            # (width, height) of the window, worked out from the screen
    mode, muted, fullscreen = 1, False, False
    text = ""                 # Letters mode: typed text
    history = []              # Phrases mode: last phrases
    banner, banner_until = "", 0.0
    wrong_until = 0.0

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            # hand tracking runs on the camera's own picture (no distortion) ...
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # ... then the picture is scaled to fit this computer's screen
            if display is None:
                sw, sh = screen_size()
                display = fit_window(frame.shape[1] / frame.shape[0], (sw, sh))
                cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(TITLE, *display)
                cv2.moveWindow(TITLE, max(0, (sw - display[0]) // 2), max(0, (sh - display[1]) // 4))
            if (frame.shape[1], frame.shape[0]) != display:
                frame = cv2.resize(frame, display)
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            now = time.time()
            draw_mode_bar(frame, mode)

            if mode in (1, 3):
                info = letters.update(results.multi_hand_landmarks)
                if mode == 1:
                    if info["new"]:
                        text += info["new"]
                        if not muted:
                            speak(info["new"])
                    if info["hand"] and info["label"]:
                        put(frame, f"{info['label']}  {info['conf']:.0%}", (10, 100), 2.2, GREEN, 4)
                    else:
                        put(frame, "Show your hand", (10, 80), 0.9, GREY)
                    put(frame, "Text: " + text[-24:], (10, 450), 1.1, WHITE)
                    put(frame, "R reset   SPACE space   S speak", (10, 474), 0.6, GREY, 1)
                else:  # Game
                    if info["new"]:
                        result = game.feed(info["new"])
                        if result == "done":
                            banner, banner_until = f"Great! {game.word_done}", now + 1.5
                            if not muted:
                                speak(game.word_done)
                            game.new_word()
                        elif result == "wrong":
                            wrong_until = now + 1.0
                    draw_word(frame, game.word, game.pos, 170)
                    if now < banner_until:
                        put_centered(frame, banner, 250, 1.5, GREEN, 3)
                    elif now < wrong_until:
                        put_centered(frame, "Try again", 250, 1.2, RED, 3)
                    elif not info["hand"]:
                        put_centered(frame, "Show your hand", 250, 1.0, GREY)
                    else:
                        put_centered(frame, "Sign the yellow letter", 250, 1.0, GREY)
                    put(frame, f"Words spelled: {game.score}", (10, 450), 1.3, YELLOW, 3)
                    hint = "R reset   N skip"
                    dw = frame.shape[1] / unit(frame)
                    put(frame, hint, (dw - text_w(frame, hint, 0.7, 2) - 10, 450), 0.7, GREY, 2)
            else:  # Phrases
                if not phrases.available:
                    put(frame, "No phrase model: run collect.py, then train.py", (10, 80), 0.8, RED)
                else:
                    info = phrases.update(results, now)
                    if info["spoken"]:
                        history = (history + [info["spoken"]])[-4:]
                        if not muted:
                            speak(info["spoken"])
                    if info["guess"]:
                        put(frame, f"{info['guess']} ({info['conf']:.0%})", (10, 90), 1.4, GREEN, 3)
                    elif info["hand_recent"]:
                        put(frame, "...", (10, 90), 1.4, GREY, 3)
                    else:
                        put(frame, "Show your hand", (10, 80), 0.9, GREY)
                    for n, phrase in enumerate(reversed(history)):
                        put(frame, phrase, (10, 455 - 40 * n), 1.1, WHITE)

            if muted:
                dw = frame.shape[1] / unit(frame)
                put(frame, "MUTED", (dw - 110, 28), 0.7, RED)
            cv2.imshow(TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("1"), ord("2"), ord("3")):
                mode = int(chr(key))
                letters.reset()
                phrases.reset()
            elif key == ord("m"):
                muted = not muted
            elif key == ord("c"):
                text, history = "", []
            elif key == ord("r"):
                if mode == 1:            # Letters: clear the typed text
                    text = ""
                    letters.reset()
                elif mode == 3:          # Game: score back to 0, new word
                    game.reset()
                    banner_until = wrong_until = 0.0
            elif key == ord("n") and mode == 3:
                game.skip()
            elif key == ord(" ") and mode == 1:
                text += " "
            elif key == ord("s") and mode == 1 and text.strip() and not muted:
                speak(text)
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    TITLE, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
