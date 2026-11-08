"""common.py - shared helpers so training and the live app prepare clips identically."""
import numpy as np

MODEL_FILE = "model.joblib"
IDLE_LABEL = "(no gesture)"  # must match the last entry of PHRASES in collect.py

# (shape_start, shape_end, wrist_start) for the Left and Right hand slots
# of the 130-number frame layout produced by collect.frame_features().
HANDS = [(0, 63, 63), (65, 128, 128)]


def preprocess(clip):
    """Make wrist movement relative to where the hand was when it first appeared.

    That way the same gesture is recognised wherever in the frame you start it.
    Frames where a hand is absent get a wrist value of 0.
    """
    out = clip.copy()
    for s, e, w in HANDS:
        present = np.abs(clip[:, s:e]).sum(axis=1) > 0
        if present.any():
            first = clip[np.argmax(present), w:w + 2]
            out[:, w:w + 2] = np.where(present[:, None], clip[:, w:w + 2] - first, 0.0)
        else:
            out[:, w:w + 2] = 0.0
    return out


def clip_to_vector(clip):
    """Turn a (N_FRAMES, 130) clip into one flat feature vector for the classifier."""
    return preprocess(clip).flatten()
