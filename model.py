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

