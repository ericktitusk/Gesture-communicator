"""phrases.py - custom gesture phrases (poses and movements), from Gesture-communicator.

Same logic as the original app.py, wrapped in a class so it can share the camera
loop with the letters mode. Needs collect.py, common.py, train.py and model.joblib.
"""
from collections import deque

import joblib
import numpy as np

from collect import DURATION, N_FRAMES, frame_features, resample
from common import IDLE_LABEL, MODEL_FILE, clip_to_vector

THRESH = 0.70    # minimum confidence to accept a gesture
CONFIRM = 3      # consecutive confident predictions needed before speaking
COOLDOWN = 2.0   # seconds of silence after speaking
STEP = 0.2       # seconds between predictions


class PhraseRecognizer:
    def __init__(self):
        try:
            self.model = joblib.load(MODEL_FILE)
        except FileNotFoundError:
            self.model = None
        self.reset()

    @property
    def available(self):
        return self.model is not None

    def reset(self):
        self.buf = deque()  # (time, features, has_hand)
        self.streak_label, self.streak = None, 0
        self.last_pred, self.cooldown_until = 0.0, 0.0
        self.last_spoken, self.rearmed = None, True
        self.guess, self.conf = "", 0.0

    def update(self, results, now):
        """Returns dict: guess, conf, hand_recent, spoken (phrase just confirmed or None)."""
        spoken = None
        feats, has_hand = frame_features(results)
        self.buf.append((now, feats, has_hand))
        while self.buf and now - self.buf[0][0] > DURATION:
            self.buf.popleft()

        hand_recent = any(h for t, _, h in self.buf if now - t < 0.5)
        window_full = self.buf[-1][0] - self.buf[0][0] >= 0.9 * DURATION

        if not hand_recent:
            self.streak_label, self.streak, self.guess, self.conf = None, 0, "", 0.0
            self.rearmed = True
        elif self.model is not None and window_full and now - self.last_pred >= STEP:
            self.last_pred = now
            clip = resample([t for t, _, _ in self.buf],
                            [f for _, f, _ in self.buf], N_FRAMES)
            proba = self.model.predict_proba([clip_to_vector(clip)])[0]
            i = int(np.argmax(proba))
            label, self.conf = self.model.classes_[i], float(proba[i])
            self.guess = label if (label != IDLE_LABEL and self.conf >= THRESH) else ""

            if not self.guess:
                self.streak_label, self.streak = None, 0
                self.rearmed = True
            elif now >= self.cooldown_until and (
                    self.guess != self.last_spoken or self.rearmed):
                if self.guess == self.streak_label:
                    self.streak += 1
                else:
                    self.streak_label, self.streak = self.guess, 1
                if self.streak >= CONFIRM:
                    spoken = self.guess
                    self.last_spoken, self.rearmed = self.guess, False
                    self.cooldown_until = now + COOLDOWN
                    self.streak_label, self.streak = None, 0

        return dict(guess=self.guess, conf=self.conf,
                    hand_recent=hand_recent, spoken=spoken)
