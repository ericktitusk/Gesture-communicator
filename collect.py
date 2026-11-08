"""
collect.py - record gesture CLIPS (poses and movements) for the Hand Gesture Communicator.

Setup (Python 3.10-3.12, Mac or Windows):
    pip install mediapipe opencv-python numpy

Controls (click the video window first):
    1-9    choose which phrase to record
    SPACE  record one clip: 1 s countdown, then 2 s of recording
    u      undo (delete) the last clip of the selected phrase
    q      quit

Each clip is resampled to N_FRAMES frames, so it does not matter how fast
your camera or computer is. Clips are saved as data/<phrase>/<timestamp>.npy
with shape (N_FRAMES, 130), so you can stop and resume any time.
Edit PHRASES below to change your gesture set (max 9).
Keep "(no gesture)" last: record resting hands, random movement and
transitions there, so the live app learns when NOT to speak.
"""
import glob
import os
import re
import time

import cv2
import mediapipe as mp
import numpy as np

PHRASES = ["Good Morning", "How are you?", "I am fine", "I need help", "(no gesture)"]
TARGET = 40       # clips to record per phrase
COUNTDOWN = 1.0   # seconds to get ready
DURATION = 2.0    # seconds per clip
N_FRAMES = 30     # frames per clip after resampling
DATA_DIR = "data"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def phrase_dir(name):
    return os.path.join(DATA_DIR, slug(name))


def count_clips(name):
    return len(glob.glob(os.path.join(phrase_dir(name), "*.npy")))


def frame_features(results):
    """130 numbers per frame: for each hand (Left, Right) 63 shape values
    + 2 raw wrist coordinates. Shape values are shifted to the wrist and
    scaled, so they ignore position and distance. The raw wrist x,y keeps
    the movement information. A missing hand is filled with zeros.
    """
    slots = {"Left": np.zeros(65), "Right": np.zeros(65)}
    has_hand = bool(results.multi_hand_landmarks)
    if has_hand:
        for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handed.classification[0].label
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            wrist = pts[0, :2].copy()
            pts -= pts[0]
            scale = np.abs(pts).max() or 1.0
            slots[label] = np.concatenate([(pts / scale).flatten(), wrist])
    return np.concatenate([slots["Left"], slots["Right"]]), has_hand


def resample(times, feats, n):
    times, feats = np.array(times), np.array(feats)
    grid = np.linspace(times[0], times[-1], n)
    return np.stack([np.interp(grid, times, feats[:, j]) for j in range(feats.shape[1])], axis=1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the camera. Check camera permissions.")
        return

    current = 0
    state, t0 = "idle", 0.0
    buf_t, buf_f, hand_frames = [], [], 0
    message = "Pick a phrase (number key), then press SPACE to record a clip"

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            now = time.time()
            feats, has_hand = frame_features(results)

            if state == "countdown" and now - t0 >= COUNTDOWN:
                state, t0 = "recording", now
                buf_t, buf_f, hand_frames = [], [], 0
            elif state == "recording":
                buf_t.append(now - t0)
                buf_f.append(feats)
                hand_frames += has_hand
                if now - t0 >= DURATION:
                    state = "idle"
                    name = PHRASES[current]
                    is_idle_class = name == "(no gesture)"
                    if len(buf_t) < 2 or (not is_idle_class and hand_frames < 0.5 * len(buf_t)):
                        message = "Hand not seen enough - clip discarded, try again"
                    else:
                        clip = resample(buf_t, buf_f, N_FRAMES).astype(np.float32)
                        os.makedirs(phrase_dir(name), exist_ok=True)
                        np.save(os.path.join(phrase_dir(name), f"{int(time.time() * 1000)}.npy"), clip)
                        message = f"Saved clip {count_clips(name)} for '{name}'"

            # --- on-screen display ---
            if state == "countdown":
                left = COUNTDOWN - (now - t0)
                cv2.putText(frame, f"Get ready... {left:.1f}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            elif state == "recording":
                cv2.putText(frame, "RECORDING", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                w = int(frame.shape[1] * min((now - t0) / DURATION, 1.0))
                cv2.rectangle(frame, (0, 45), (w, 55), (0, 0, 255), -1)
            else:
                cv2.putText(frame, f"Selected: {PHRASES[current]}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

            for i, name in enumerate(PHRASES):
                mark = ">" if i == current else " "
                cv2.putText(frame, f"{mark}{i + 1}. {name}: {count_clips(name)}/{TARGET}",
                            (10, 85 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, message, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

            cv2.imshow("Collect gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if state != "idle":
                continue
            if key == ord(" "):
                state, t0 = "countdown", now
            elif key == ord("u"):
                files = sorted(glob.glob(os.path.join(phrase_dir(PHRASES[current]), "*.npy")))
                if files:
                    os.remove(files[-1])
                    message = f"Deleted last clip of '{PHRASES[current]}'"
            elif ord("1") <= key < ord("1") + len(PHRASES):
                current = key - ord("1")
                message = f"Selected '{PHRASES[current]}'. Press SPACE to record"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
