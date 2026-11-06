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

# Step 3 - entity_lists
import numpy as np
from collections import Counter

CAST = ("lily", "timmy", "ben", "tom", "max", "tim", "sam", "jack", "lucy", "anna", "mia", "sarah", "jane", "bob", "john",
        "sue", "benny", "sally", "joe", "billy", "sara", "tommy", "emma", "amy", "sammy", "molly", "jill", "lila", "jake",
        "jenny", "emily", "jimmy", "jen", "mary", "lola", "peter", "johnny", "jim", "pete", "ella", "kim", "mandy", "bella",
        "leo", "alex", "ted", "dave", "lisa", "mike", "henry", "jerry", "susie", "ann", "andy", "charlie", "tony", "suzy",
        "clara", "alice", "fred", "lilly", "steve", "maria", "bobby", "harry", "james", "joey", "millie", "sophie", "ellie",
        "thomas", "toby", "matt", "george", "polly", "zoe", "paul", "julie", "suzie", "chloe", "kate", "jeff", "chris")

SURNAMES = ("green", "brown", "black", "white", "gray", "fox", "bird", "wolf", "bear", "hill", "wood", "stone", "lake", "king",
            "baker", "cook", "hunter", "farmer", "bell", "rose", "snow", "day", "long", "young", "little", "pond", "field")

PLACE_A = ("new", "old", "north", "south", "east", "west", "green", "oak", "silver", "red", "blue", "big", "little", "long",
           "high", "sun", "fair", "rock", "clear", "white")

PLACE_B = ("york", "town", "hill", "lake", "city", "valley", "river", "bay", "falls", "port", "field", "wood", "land", "beach",
           "island", "harbor", "ridge", "creek", "forest", "spring")


def entity_lists(k=100, seed=0):
    # Load the first 3000 TinyStories and count the frequency of each token.
    stories = load_stories(0, 3000)

    token_counts = Counter(
        token
        for story in stories
        for sentence in story
        for token in sentence
    )

    # Identify common words, excluding the existing TinyStories character names.
    cast_set = set(CAST)

    common = {
        word
        for word, count in token_counts.items()
        if count >= 3 and word not in cast_set
    }

    # Collect first names from the names registry in its original order.
    # Exclude common words, surnames, and place-name components.
    excluded = common | set(SURNAMES) | set(PLACE_A) | set(PLACE_B)

    registry_names = dml_data.load("names").names

    names = []

    for name in registry_names:
        name = name.lower()

        if len(names) >= k:
            break

        if len(name) > 2 and name not in excluded:
            names.append(name)

    # Generate all 400 place combinations in nested order.
    places = [
        (a, b)
        for a in PLACE_A
        for b in PLACE_B
    ]

    # Initialize the random number generator for reproducibility.
    rng = np.random.default_rng(seed)

    # Split first names into labeled and held-out halves.
    name_idx = rng.permutation(len(names))

    train_names = [names[i] for i in name_idx[::2]]
    heldout_names = [names[i] for i in name_idx[1::2]]

    # Split places using a second independent permutation from the same RNG.
    place_idx = rng.permutation(len(places))

    train_places = [places[i] for i in place_idx[::2]]
    heldout_places = [places[i] for i in place_idx[1::2]]

    # Return the labeled, held-out, and combined entity lists.
    return {
        "train": (train_names, train_places),

        "heldout": (heldout_names, heldout_places),

        "all": (
            train_names + heldout_names,
            train_places + heldout_places
        )
    }

