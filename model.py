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

# Step 4 - plant_entities
TAGS = ("O", "B-Per", "I-Per", "B-Loc", "I-Loc")

PLACE_NOUNS = (
    "park", "store", "forest", "beach", "woods", "river", "zoo", "lake",
    "jungle", "town", "city", "farm", "garden", "library", "hospital",
    "market", "meadow", "mountain", "ocean", "pond", "shop", "castle",
    "airport", "field"
)

PLACE_PREPS = ("to", "at", "in", "near", "from", "by")


def plant_entities(stories, first_names, places, seed=0,
                   place_rate=0.8, append_rate=0.3):

    # Use one random number generator for the entire function call.
    rng = np.random.default_rng(seed)

    planted_stories = []

    # Process each story independently.
    for story in stories:

        # Each story gets its own character and place mappings.
        person_mentions = {}
        place_mentions = {}
        home_place = None

        planted_story = []

        # Process each sentence from left to right.
        for sentence in story:

            out_tokens = []
            out_tags = []

            i = 0

            while i < len(sentence):

                word = sentence[i]

                # -------------------------------------------------
                # 1. PLACE REPLACEMENT
                # -------------------------------------------------

                # Detect the pattern: preposition + "the" + place noun.
                if (
                    word == "the"
                    and i > 0
                    and i + 1 < len(sentence)
                    and sentence[i - 1] in PLACE_PREPS
                    and sentence[i + 1] in PLACE_NOUNS
                ):
                    noun = sentence[i + 1]

                    # Assign an identity to this place noun the first
                    # time it appears in the current story.
                    if noun not in place_mentions:

                        if rng.random() < place_rate:
                            place_idx = int(rng.integers(len(places)))
                            place_mentions[noun] = places[place_idx]
                        else:
                            place_mentions[noun] = None

                    place = place_mentions[noun]

                    # Replace "the" and the noun with the selected place.
                    if place is not None:

                        out_tokens.extend(place)
                        out_tags.extend([3, 4])

                        # Skip both "the" and the original place noun.
                        i += 2
                        continue

                    # If the noun has no assigned place, leave the
                    # original words unchanged.

                # -------------------------------------------------
                # 2. PEOPLE REPLACEMENT
                # -------------------------------------------------

                # Detect possessive character names such as "lily's".
                possessive = (
                    word.endswith("'s")
                    and word[:-2] in CAST
                )

                character = word[:-2] if possessive else word

                # A character can be a direct cast token or possessive.
                if character in CAST:

                    # Choose a mention only on the first occurrence
                    # of this character in the current story.
                    if character not in person_mentions:

                        choice = rng.random()

                        if choice < 0.45:
                            # 45%: First name only.
                            mention = [
                                first_names[
                                    int(rng.integers(len(first_names)))
                                ]
                            ]

                        elif choice < 0.70:
                            # 25%: First name followed by surname.
                            first = first_names[
                                int(rng.integers(len(first_names)))
                            ]

                            surname = SURNAMES[
                                int(rng.integers(len(SURNAMES)))
                            ]

                            mention = [first, surname]

                        else:
                            # 30%: Mr/Mrs followed by surname.
                            title = ("mr", "mrs")[
                                int(rng.integers(2))
                            ]

                            surname = SURNAMES[
                                int(rng.integers(len(SURNAMES)))
                            ]

                            mention = [title, surname]

                        person_mentions[character] = mention

                    # Reuse the same mention throughout this story.
                    mention = person_mentions[character]

                    out_tokens.extend(mention)

                    # First mention token receives B-Per (1).
                    # Remaining mention tokens receive I-Per (2).
                    out_tags.extend(
                        [1] + [2] * (len(mention) - 1)
                    )

                    # Preserve possessive suffixes with tag O (0).
                    if possessive:
                        out_tokens.append("'s")
                        out_tags.append(0)

                else:
                    # Copy every other token unchanged with tag O.
                    out_tokens.append(word)
                    out_tags.append(0)

                i += 1

            # -----------------------------------------------------
            # 3. APPEND THE STORY'S HOME PLACE
            # -----------------------------------------------------

            # Append a home place only if:
            # - The sentence contains a person (B-Per).
            # - The sentence contains no place (B-Loc).
            # - There are at least 3 tokens.
            # - The sentence ends with ., ! or ?.
            if (
                1 in out_tags
                and 3 not in out_tags
                and len(out_tokens) >= 3
                and out_tokens[-1] in {".", "!", "?"}
                and rng.random() < append_rate
            ):

                # Draw one home place per story, only when needed.
                if home_place is None:
                    home_idx = int(rng.integers(len(places)))
                    home_place = places[home_idx]

                # Select one of the allowed appended prepositions.
                prep_options = ("at", "in", "near")

                prep = prep_options[int(rng.integers(3))]

                # Insert the preposition and home place before
                # the final punctuation mark.
                out_tokens[-1:-1] = [
                    prep,
                    home_place[0],
                    home_place[1]
                ]

                out_tags[-1:-1] = [0, 3, 4]

            # Store the processed sentence as (tokens, tags).
            planted_story.append((out_tokens, out_tags))

        planted_stories.append(planted_story)

    return planted_stories

