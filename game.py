"""game.py - very simple spelling game: a random 3-letter word, sign its letters, count the successes.

Words are short (3 letters), avoid J and Z (they need motion) and double letters
(a held letter is only counted once until the hand changes or leaves the frame).
"""
import random

WORDS = [
    "CAT", "DOG", "SUN", "BOY", "BUS", "BED", "CUP", "PIG", "COW", "HAT",
    "BAG", "BOX", "FOX", "LOG", "WEB", "KID", "RED", "LID", "FLY", "CAB",
    "FIG", "PEN", "BIG", "TOP", "MAP",
]


class WordGame:
    def __init__(self, words=WORDS):
        self.words = list(words)
        self.score = 0          # words spelled successfully
        self.word = ""
        self.pos = 0            # index of the letter to sign next
        self.word_done = ""     # the word just completed
        self.new_word()

    def new_word(self):
        choices = [w for w in self.words if w != self.word] or self.words
        self.word = random.choice(choices)
        self.pos = 0

    def feed(self, letter):
        """Give a confirmed letter. Returns 'correct', 'wrong' or 'done'."""
        if letter == self.word[self.pos]:
            self.pos += 1
            if self.pos == len(self.word):
                self.score += 1
                self.word_done = self.word
                return "done"
            return "correct"
        return "wrong"

    def skip(self):
        self.new_word()

    def reset(self):
        self.score = 0
        self.word = ""
        self.new_word()
