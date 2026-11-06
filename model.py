"""
Pretrain a Tiny BERT, Then Fine-Tune Four Heads

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - split_sentences
import re

TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?]")

def split_sentences(text):
    # Convert the input text to lowercase and extract valid tokens.
    tokens = TOKEN.findall(text.lower())

    sentences = []
    current_sentence = []

    for token in tokens:
        current_sentence.append(token)

        # End the current sentence when '.', '!' or '?' is encountered.
        if token in {".", "!", "?"}:
            sentences.append(current_sentence)
            current_sentence = []

    # Preserve any remaining tokens that do not end with punctuation.
    if current_sentence:
        sentences.append(current_sentence)

    return sentences

# Step 2 - load_stories
import dml_data

_DATA = {}

def load_stories(start=0, stop=1000):
    # Load and cache TinyStories only once.
    if "stories" not in _DATA:
        _DATA["stories"] = dml_data.load("tinystories").stories

    # Select the requested slice of stories.
    stories = _DATA["stories"][start:stop]

    # Tokenize each story into a list of sentences.
    return [split_sentences(story) for story in stories]

