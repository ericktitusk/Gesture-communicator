"""letters.py - static ASL letters (A-Y, no J/Z) from 21 hand landmarks.

Uses the Keras model AmimulBmeIU/asl-alphabet-classifier (input: 21 landmarks x
(x, y, z) flattened to 63 numbers). The model and labels are downloaded once into
./models and then work offline.
"""
import json
import os
from collections import Counter, deque

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

REPO = "AmimulBmeIU/asl-alphabet-classifier"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FILES = ("model.keras", "labels.json")

WINDOW = 12      # predictions voted over (about half a second)
STABLE = 0.75    # share of the window that must agree before a letter counts

# The model was trained on landmarks from an un-mirrored camera image. The app shows a
# mirrored image, so x is flipped back. If letters look wrong, try False.
UNMIRROR_X = True


def _ensure_files():
    paths = {name: os.path.join(MODEL_DIR, name) for name in FILES}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    from huggingface_hub import hf_hub_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name in FILES:
        hf_hub_download(REPO, name, local_dir=MODEL_DIR)
    return paths


class LetterRecognizer:
    def __init__(self):
        import keras
        paths = _ensure_files()
        self.model = keras.saving.load_model(paths["model.keras"])
        with open(paths["labels.json"]) as f:
            labels = json.load(f)
        if isinstance(labels, dict):  # {"0": "A", ...}
            labels = [labels[k] for k in sorted(labels, key=int)]
        self.labels = labels
        self.reset()

    def reset(self):
        self.recent = deque(maxlen=WINDOW)
        self.last_accepted = None

    def update(self, multi_hand_landmarks):
        """Returns dict: hand, label (smoothed), conf, stability, new (letter just confirmed)."""
        if not multi_hand_landmarks:
            self.reset()
            return dict(hand=False, label=None, conf=0.0, stability=0.0, new=None)

        lm = multi_hand_landmarks[0].landmark
        feats = np.array(
            [[(1.0 - p.x) if UNMIRROR_X else p.x, p.y, p.z] for p in lm],
            dtype="float32",
        ).reshape(1, -1)
        probs = np.asarray(self.model(feats, training=False))[0]
        i = int(np.argmax(probs))
        self.recent.append(str(self.labels[i]))

        label, votes = Counter(self.recent).most_common(1)[0]
        stability = votes / len(self.recent)
        new = None
        if (len(self.recent) == WINDOW and stability >= STABLE
                and label != self.last_accepted):
            self.last_accepted = label
            new = label
        return dict(hand=True, label=label, conf=float(probs[i]),
                    stability=stability, new=new)
