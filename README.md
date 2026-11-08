# SVS Symposium 2026 - ASL & Gesture Communicator

A desktop app that watches your hands through a webcam and turns what it sees into text and speech.
It has three modes:

| Key | Mode | What it does |
|-----|------|--------------|
| `1` | **Letters** | Recognizes the static ASL (American Sign Language) alphabet, A-Y. Each steady letter is added to a text line and spoken aloud. |
| `2` | **Phrases** | Recognizes your own recorded gestures, including movements such as waving, and speaks the phrase (for example "Good Morning"). |
| `3` | **Game** | A random 3-letter word appears. Sign its letters in order. The screen counts how many words you spelled. |

It runs on **macOS and Windows**, entirely on your own computer. After the first start it needs no internet.

---

## 1. How to run it

### What you need
- A computer with a webcam, running macOS or Windows
- **Python 3.10, 3.11 or 3.12** (MediaPipe does not support newer versions). Check with `python --version`.
- Internet for the **first** run only (to install packages and download the letter model)

### Steps

```
# 1. Get the project
git clone https://github.com/ericktitusk/gesture-communicator.git
cd gesture-communicator

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS
venv\Scripts\activate             # Windows

# 3. Install the packages
pip install -r requirements.txt

# 4. Start the app
python app.py
```

- **First start:** the app looks for the ASL letter model in the `models/` folder. If it is not there, it downloads it once. Starting takes a few seconds because the model has to load.
- **macOS:** allow camera access for Terminal (or your editor) when asked. If you missed the prompt, go to System Settings > Privacy & Security > Camera.
- Press `q` to quit.

### Keys

| Key | Action |
|-----|--------|
| `1` / `2` / `3` | Switch mode: Letters / Phrases / Game |
| `m` | Mute or unmute speech |
| `r` | Reset. Letters: clear the typed text. Game: score back to 0 and a new word |
| `n` | Game: skip to the next word |
| `space` | Letters: add a space |
| `s` | Letters: speak the whole typed text |
| `c` | Clear the typed text and the phrase history |
| `f` | Full screen on/off |
| `q` | Quit |

The window size is worked out from your screen automatically. You can also drag it bigger.

### Tips for a good demo
- Good, even light on your hand. Avoid a bright window behind you.
- Keep one hand clearly in view, about an arm's length from the camera.
- **Letters and Game:** hold each letter steady for about half a second. Letters are counted once; to repeat the same letter, lower your hand or change letter first.
- **Phrases:** do the gesture inside the 2-second window and hold it. To repeat the same phrase, lower your hand first.
- J and Z are not supported, because in ASL they are drawn with a movement.

---

## 2. How it is built

### The big picture

```
 Webcam
   |
   v
 MediaPipe Hands          finds 21 points (landmarks) on each hand, every frame
   |
   +--> Letters mode  --> neural network  --> letter A-Y  --> text, speech
   |
   +--> Phrases mode  --> 2-second clip --> SVM classifier --> phrase --> text, speech
   |
   +--> Game mode     --> same letter recognizer --> checks the next letter of the word
```

Everything starts from the same step: the computer does not look at your hand as a picture. It first finds **21 points** on the hand (finger tips, knuckles, the wrist). The rest of the app only uses those points, which is why it is fast and works on an ordinary laptop.

### Step by step

1. **Hand tracking (MediaPipe).** Google's MediaPipe library finds the 21 landmarks on up to two hands in every video frame.
2. **Letters mode.** The 21 points (x, y, z each, 63 numbers) go into a small neural network trained to recognize the static ASL alphabet. The model is `AmimulBmeIU/asl-alphabet-classifier` from Hugging Face. Its author reports about 94% accuracy on its own test data. To avoid flicker, the app looks at the last 12 guesses and only accepts a letter when at least 75% of them agree. The accepted letter is added to the text and spoken.
3. **Phrases mode.** A gesture is a short **2-second clip**, resampled to 30 frames so camera speed does not matter. Each frame stores the hand shape and the wrist position, which carries the movement. A scikit-learn **SVM classifier** is trained on clips you record yourself. In the live app, the last 2 seconds of hand data are checked about 5 times a second, and a phrase is spoken only after several confident guesses in a row.
4. **Game mode.** It uses the same letter recognizer as Letters mode. The game picks a random word from a list of 25 short words, highlights the next letter, and moves on when that letter is recognized. When the whole word is done, the score goes up by one and the word is spoken.
5. **Speech.** macOS uses its built-in `say` command. Windows uses `pyttsx3` (the system voices). It runs offline.

### Project files

```
app.py            Main program: camera, window, the three modes, keys
letters.py        ASL letter recognizer (loads the model, smooths the guesses)
phrases.py        Phrase recognizer (clips + SVM), from the original Gesture Communicator
game.py           The spelling game (word list, score)
speech.py         Text to speech for Mac and Windows
collect.py        Record your own gesture clips
train.py          Train the phrase classifier from the recorded clips
common.py         Clip preparation shared by training and the app
model.joblib      The trained phrase classifier
models/           The ASL letter model (model.keras and labels.json)
requirements.txt  Python packages and their versions
```

### Technologies used

| Part | Tool |
|------|------|
| Hand tracking | MediaPipe Hands (Google) |
| Camera and window | OpenCV |
| Letter model | Keras / TensorFlow (model from Hugging Face) |
| Phrase model | scikit-learn (SVM), joblib |
| Speech | macOS `say`, Windows `pyttsx3` |
| Language | Python |

### Credits
- ASL letter model: [AmimulBmeIU/asl-alphabet-classifier](https://huggingface.co/AmimulBmeIU/asl-alphabet-classifier) on Hugging Face
- Hand tracking: [MediaPipe](https://ai.google.dev/edge/mediapipe) by Google

---

## 3. Teaching the app your own phrases

The phrase recognizer only knows the gestures you record. To change them:

1. Edit the `PHRASES` list at the top of `collect.py` (maximum 9). Keep `"(no gesture)"` as the last entry.
2. Record the gestures:

   ```
   python collect.py
   ```

   | Key | Action |
   |-----|--------|
   | `1`-`9` | Select a phrase |
   | `SPACE` | Record one clip (1 second countdown, then 2 seconds recording) |
   | `u` | Delete the last clip of the selected phrase |
   | `q` | Quit |

   Record about 40 clips per phrase, and vary speed, distance, hand position and lighting a little. Use **(no gesture)** to record resting hands, random movement, and hands entering or leaving the frame. This teaches the app when *not* to speak. Clips are saved in a `data/` folder.

3. Train:

   ```
   python train.py
   ```

   This prints the accuracy for each phrase and saves a new `model.joblib`. Accuracy measured on clips from one sitting looks better than it really is, so also test live, ideally with a different person.

4. Run `python app.py` and press `2`.

The recorded clips (`data/`) are not stored in this repository. `model.joblib` is the trained result, so the app works without them.

---

## 4. Settings you can change

| File | Setting | Default | Effect |
|------|---------|---------|--------|
| `phrases.py` | `THRESH` | 0.70 | Minimum confidence to accept a phrase. Raise it if you get false triggers. |
| `phrases.py` | `CONFIRM` | 3 | Confident guesses in a row needed before speaking. |
| `phrases.py` | `COOLDOWN` | 2.0 | Seconds of silence after speaking. |
| `letters.py` | `WINDOW` | 12 | Number of recent guesses that are compared. |
| `letters.py` | `STABLE` | 0.75 | Share of those guesses that must agree before a letter counts. |
| `letters.py` | `UNMIRROR_X` | `True` | The window shows a mirror image but the letter model was trained on an un-mirrored one. Try `False` if letters come out wrong. |
| `game.py` | `WORDS` | 25 words | The word list. Avoid J, Z and double letters. |

---

## 5. Troubleshooting

- **`module 'mediapipe' has no attribute 'solutions'`:** a newer MediaPipe was installed. Run `pip install -r requirements.txt` again, or `pip install "mediapipe==0.10.14"`.
- **MediaPipe or TensorFlow will not install:** check `python --version`. It must be 3.10, 3.11 or 3.12.
- **"Could not open the camera":** close other apps that use the camera, check the camera permission, or try `cv2.VideoCapture(1)` in `app.py` if you have more than one camera.
- **Letters are often wrong:** improve the light, keep the hand fully in view, and try `UNMIRROR_X = False` in `letters.py`. Some letters (M, N, S, T, A, E) look alike and are confused more often.
- **"No phrase model":** `model.joblib` is missing. Run `collect.py` and then `train.py`. Letters and Game still work.
- **The letter model download fails:** check your internet connection. It is only needed once, then the model is kept in `models/`.
- **No speech on Windows:** make sure `pyttsx3` is installed and a system voice is available.

---

## 6. Explaining it at the symposium

### In one minute (simple version)

> "My project teaches a computer to understand hand signs.
> The camera can't understand a hand by itself, so first the computer finds 21 dots on my hand: the tips of my fingers, my knuckles and my wrist.
> Then an AI model, which learned from thousands of examples, looks at where the dots are and decides which letter of the sign language alphabet I am making.
> It also learns gestures that I teach it myself, like waving for 'Good morning'.
> When it is sure, it writes the word on the screen and says it out loud.
> And there is a game: it gives me a word, and I spell it with my hands."

### Suggested demo (3 minutes)
1. Press `1`. Sign two or three letters of your name. Point out the text line and the spoken letter.
2. Press `2`. Show one or two phrases you recorded. Explain that you taught it these.
3. Press `3`. Spell one word in the game. Show the score going up. Press `r` to reset for the next visitor.

### Questions people may ask
- **Is it always right?** No. The letter model is about 94% right on its own test data, and less in real life with poor light or a different hand. Some letters look alike.
- **Why is the camera picture not used directly?** Using 21 points instead of the whole picture makes it faster and works the same for different skin tones, backgrounds and clothes.
- **How does it learn?** It is shown many examples with the correct answer and slowly adjusts until its guesses match the answers. This is called *training*.
- **Does it need the internet?** Only the first time, to download the model. After that it works offline.
- **Can it do all of sign language?** No. It knows the static alphabet (not J and Z, which need movement) and the few phrases that were recorded. Real sign language has thousands of signs, uses both hands, the face and the body, and has its own grammar.
- **Where does the letter model come from?** It was trained by another developer (AmimulBmeIU) and shared on Hugging Face. This project uses it and adds the phrases, the game and the desktop app.

### Checklist for symposium day
- [ ] Run the app once on the laptop you will use, with the same lighting as the hall if possible
- [ ] Allow camera access (macOS) and test speech volume
- [ ] Check the `models/` folder is present so the app starts without internet
- [ ] Test each mode: Letters, Phrases, Game
- [ ] Have a backup: the Streamlit web demo, or a short screen recording of the app working

---

## Limitations
- The letter model covers A-Y only (no J, Z) and works best with one hand.
- Phrases are limited to the gestures you record, over a 2-second window, one at a time.
- Accuracy depends on light and on the person. A model trained on one person is less accurate for others, so record phrase clips from several people if many will use it.
