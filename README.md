# SVS Symposium 2026 - ASL & Gesture Communicator

One desktop app, three modes (press the number key):

1. **Letters** - static ASL letters A-Y (no J/Z). Held letters are added to a text line and spoken.
2. **Phrases** - your own recorded gestures (poses and movements), spoken aloud.
3. **Game** - a random 3-letter word appears; sign its letters in order. Counts words spelled.

## Setup
Copy `collect.py`, `train.py`, `common.py`, `data/` and `model.joblib` from Gesture-communicator
into this folder (they are used unchanged), then:

    pip install -r requirements.txt
    python app.py

The first run downloads the letter model into `models/` (internet needed once).
Record and train phrases exactly as before: `python collect.py`, then `python train.py`.

The window size is worked out from your screen automatically. Drag it bigger, or press `f` for full screen.

## Keys
1/2/3 mode - m mute - c clear text/phrase history - space add space (Letters)
s speak the text (Letters) - r reset (Letters: clear text, Game: score to 0) - n next word (Game) - f full screen - q quit
