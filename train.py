"""
train.py - train the gesture classifier from the clips recorded by collect.py.

Run:  python train.py
Output: model.joblib (used by app.py)

Put collect.py, common.py, train.py and app.py in the same folder.
Re-run this any time you record more clips or change PHRASES.
"""
import glob
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from collect import PHRASES, phrase_dir
from common import MODEL_FILE, clip_to_vector


def load_data():
    X, y = [], []
    for name in PHRASES:
        files = glob.glob(os.path.join(phrase_dir(name), "*.npy"))
        if not files:
            print(f"  {name}: no clips, skipped")
            continue
        print(f"  {name}: {len(files)} clips")
        for path in files:
            X.append(clip_to_vector(np.load(path)))
            y.append(name)
    return np.array(X), np.array(y)


def main():
    print("Loading clips...")
    X, y = load_data()
    if len(set(y)) < 2:
        print("Need clips for at least 2 phrases. Record more with collect.py.")
        return

    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, probability=True))

    smallest = int(np.unique(y, return_counts=True)[1].min())
    if smallest >= 3:
        folds = min(5, smallest)
        cv = StratifiedKFold(folds, shuffle=True, random_state=0)
        pred = cross_val_predict(model, X, y, cv=cv)
        print("\nCross-validation results:")
        print(classification_report(y, pred, zero_division=0))
        print(f"Cross-validated accuracy: {(pred == y).mean():.1%}")
        print("(Clips from one sitting look alike, so real-world accuracy is usually lower.)")
    else:
        print("\nToo few clips for an accuracy check (need at least 3 per phrase).")

    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved {MODEL_FILE}. Now run: python app.py")


if __name__ == "__main__":
    main()
