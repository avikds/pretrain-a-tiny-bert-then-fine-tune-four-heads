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

# Step 5 - build_vocab
PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4

SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def build_vocab(stories, max_size=1200):
    # Count the frequency of every token across all stories.
    counts = Counter()

    for story in stories:
        for tokens, tags in story:
            counts.update(tokens)

    # Sort tokens by descending frequency.
    # For equal frequencies, sort alphabetically.
    sorted_words = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    # Reserve the first five positions for BERT special tokens.
    vocab = {
        token: index
        for index, token in enumerate(SPECIALS)
    }

    # Add the most frequent ordinary words without exceeding max_size.
    remaining_slots = max(0, max_size - len(SPECIALS))

    for word, count in sorted_words[:remaining_slots]:
        vocab[word] = len(vocab)

    return vocab


def encode_stories(stories, vocab):
    encoded_stories = []

    # Preserve the original story and sentence nesting.
    for story in stories:
        encoded_story = []

        for tokens, tags in story:

            # Convert words into their corresponding vocabulary indices.
            # Unknown words are mapped to UNK (index 1).
            ids = np.array(
                [vocab.get(word, UNK) for word in tokens],
                dtype=np.int32
            )

            # Convert the BIO tags into a NumPy integer array.
            tag_ids = np.array(tags, dtype=np.int32)

            # Store each sentence as a pair of equal-length arrays.
            encoded_story.append((ids, tag_ids))

        encoded_stories.append(encoded_story)

    return encoded_stories

# Step 6 - pack_pair
def pack_pair(a, b=None, max_len=32):
    # Convert inputs to lists so the original sequences remain unchanged.
    a = list(a)

    # Determine whether this is a sentence pair or a single sentence.
    is_pair = b is not None

    if is_pair:
        b = list(b)

        # A pair requires at least 3 special-token positions:
        # [CLS], [SEP], and [SEP].
        if max_len < 3:
            raise ValueError("max_len must be at least 3 for a sentence pair.")

        # Truncate the longer sequence until the pair fits.
        # Remove from sequence A when both have equal length.
        while len(a) + len(b) + 3 > max_len:
            if len(a) >= len(b):
                a.pop()
            else:
                b.pop()

        # Construct the BERT input sequence.
        ids = [CLS] + a + [SEP] + b + [SEP]

        # Segment 0 covers [CLS], A, and the first [SEP].
        # Segment 1 covers B and the final [SEP].
        segs = (
            [0] * (len(a) + 2)
            + [1] * (len(b) + 1)
        )

    else:
        # A single sentence requires [CLS] and [SEP].
        if max_len < 2:
            raise ValueError("max_len must be at least 2 for a single sentence.")

        # Keep the first tokens of A that fit.
        a = a[:max_len - 2]

        # Construct the single-sentence BERT input.
        ids = [CLS] + a + [SEP]

        # All tokens belong to segment 0.
        segs = [0] * len(ids)

    # Pad the input IDs with PAD tokens.
    padding_length = max_len - len(ids)

    ids.extend([PAD] * padding_length)

    # Padding also receives segment ID 0.
    segs.extend([0] * padding_length)

    # Return fixed-length int32 NumPy arrays.
    return (
        np.array(ids, dtype=np.int32),
        np.array(segs, dtype=np.int32)
    )

# Step 7 - init_params
import math
from tinygrad import Tensor


def _weight(a):
    return Tensor(np.asarray(a, dtype=np.float32)).contiguous().realize()


def init_params(vocab_size, d_model=64, n_layers=3, n_heads=4,
                max_len=32, seed=0):

    # Ensure that the model dimension is divisible by the number of heads.
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    d = d_model

    # ---------------------------------------------------------
    # 1. Initialize the three embedding matrices.
    # ---------------------------------------------------------

    tok = _weight(
        rng.normal(0.0, 0.1, size=(vocab_size, d_model))
    )

    seg = _weight(
        rng.normal(0.0, 0.1, size=(2, d_model))
    )

    pos = _weight(
        rng.normal(0.0, 0.1, size=(max_len, d_model))
    )

    # ---------------------------------------------------------
    # 2. Initialize the Transformer encoder blocks.
    # ---------------------------------------------------------

    blocks = []

    for _ in range(n_layers):

        block = {
            # Combined query, key and value projection.
            "qkv": _weight(
                rng.normal(
                    0.0,
                    math.sqrt(1.0 / d),
                    size=(d, 3 * d)
                )
            ),

            # Attention output projection.
            "out": _weight(
                rng.normal(
                    0.0,
                    math.sqrt(1.0 / d),
                    size=(d, d)
                )
            ),

            # First feed-forward projection.
            "f1": _weight(
                rng.normal(
                    0.0,
                    math.sqrt(2.0 / d),
                    size=(d, 2 * d)
                )
            ),

            # Second feed-forward projection.
            "f2": _weight(
                rng.normal(
                    0.0,
                    math.sqrt(1.0 / (2 * d)),
                    size=(2 * d, d)
                )
            ),

            # Feed-forward biases.
            "b1": _weight(np.zeros(2 * d)),
            "b2": _weight(np.zeros(d)),

            # First layer normalization parameters.
            "g1": _weight(np.ones(d)),
            "m1": _weight(np.zeros(d)),

            # Second layer normalization parameters.
            "g2": _weight(np.ones(d)),
            "m2": _weight(np.zeros(d)),
        }

        blocks.append(block)

    # ---------------------------------------------------------
    # 3. Initialize the next-sentence prediction head.
    # ---------------------------------------------------------

    nsp_w = _weight(
        rng.normal(
            0.0,
            math.sqrt(1.0 / d),
            size=(d, 2)
        )
    )

    nsp_b = _weight(np.zeros(2))

    # ---------------------------------------------------------
    # 4. Return all parameters and model configuration values.
    # ---------------------------------------------------------

    return {
        "tok": tok,
        "seg": seg,
        "pos": pos,
        "blocks": blocks,
        "nsp_w": nsp_w,
        "nsp_b": nsp_b,

        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "max_len": max_len,
    }

# Step 8 - embed_inputs
def embed_inputs(params, ids, segs):
    # Get the batch size and sequence length.
    batch_size, T = ids.shape

    # Get the model's embedding dimension.
    d = params["d_model"]

    # Token embeddings: (B, T, d)
    token_emb = params["tok"][ids]

    # Segment embeddings: (B, T, d)
    segment_emb = params["seg"][segs]

    # Positional embeddings: (T, d)
    # Reshape to (1, T, d) for broadcasting across the batch.
    position_emb = params["pos"][:T].reshape(1, T, d)

    # Combine token, segment, and positional embeddings.
    return token_emb + segment_emb + position_emb

# Step 9 - self_attention
def self_attention(block, x, pad, n_heads):
    # Input dimensions: B = batch size, T = sequence length, d = model dimension.
    B, T, d = x.shape

    # Dimension of each attention head.
    p = d // n_heads

    # ---------------------------------------------------------
    # 1. Compute the combined Query, Key and Value projections.
    # ---------------------------------------------------------

    # Shape: (B, T, 3*d)
    qkv = x @ block["qkv"]

    # Reshape into separate Q, K and V components.
    # Shape: (B, T, 3, n_heads, p)
    qkv = qkv.reshape(B, T, 3, n_heads, p)

    # Rearrange dimensions to separate Q, K and V.
    # Shape: (3, B, n_heads, T, p)
    qkv = qkv.permute(2, 0, 3, 1, 4)

    # Extract queries, keys and values.
    # Each has shape: (B, n_heads, T, p)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    # ---------------------------------------------------------
    # 2. Calculate scaled dot-product attention scores.
    # ---------------------------------------------------------

    # Transpose the last two dimensions of the keys.
    # Shape: (B, n_heads, p, T)
    k_transposed = k.permute(0, 1, 3, 2)

    # Calculate Q multiplied by K transpose.
    # Shape: (B, n_heads, T, T)
    scores = q @ k_transposed

    # Scale the attention scores by sqrt(p).
    scores = scores / math.sqrt(p)

    # ---------------------------------------------------------
    # 3. Apply the padding mask.
    # ---------------------------------------------------------

    # Expand the padding mask to broadcast across all heads
    # and query positions.
    # Shape: (B, 1, 1, T)
    pad_mask = pad.reshape(B, 1, 1, T)

    # Assign a large negative score to padded key positions.
    # This makes their attention probability approximately zero.
    scores = scores + pad_mask.cast(scores.dtype) * (-1e9)

    # ---------------------------------------------------------
    # 4. Apply softmax over the key positions.
    # ---------------------------------------------------------

    # Shape: (B, n_heads, T, T)
    attn_weights = scores.softmax(axis=-1)

    # Multiply attention weights by the value vectors.
    # Shape: (B, n_heads, T, p)
    context = attn_weights @ v

    # ---------------------------------------------------------
    # 5. Combine the attention heads.
    # ---------------------------------------------------------

    # Rearrange from (B, n_heads, T, p) to (B, T, n_heads, p).
    context = context.permute(0, 2, 1, 3)

    # Concatenate all attention heads.
    # Shape: (B, T, d)
    context = context.reshape(B, T, d)

    # ---------------------------------------------------------
    # 6. Apply the final output projection.
    # ---------------------------------------------------------

    output = context @ block["out"]

    return output

# Step 10 - encoder_block
def layer_norm(x, g, m):
    # Calculate the mean along the last dimension.
    # keepdim=True preserves the dimension for broadcasting.
    mu = x.mean(axis=-1, keepdim=True)

    # Calculate population variance along the last dimension.
    var = ((x - mu) ** 2).mean(axis=-1, keepdim=True)

    # Normalize and apply the learnable gain and offset.
    return (x - mu) / (var + 1e-5).sqrt() * g + m


def encoder_block(block, x, pad, n_heads):
    # ---------------------------------------------------------
    # 1. Multi-head self-attention with residual connection.
    # ---------------------------------------------------------

    attention_output = self_attention(
        block, x, pad, n_heads
    )

    # Add the residual connection and apply first layer norm.
    Z = layer_norm(
        attention_output + x,
        block["g1"],
        block["m1"]
    )

    # ---------------------------------------------------------
    # 2. Position-wise feed-forward network.
    # ---------------------------------------------------------

    # First linear transformation, bias, and GELU activation.
    F = (Z @ block["f1"] + block["b1"]).gelu()

    # Second linear transformation and bias.
    F = F @ block["f2"] + block["b2"]

    # ---------------------------------------------------------
    # 3. Second residual connection and layer normalization.
    # ---------------------------------------------------------

    E = layer_norm(
        F + Z,
        block["g2"],
        block["m2"]
    )

    return E

# Step 11 - encode
def encode(params, ids, segs):
    # Identify padded positions using the PAD token ID.
    # Shape: (B, T)
    pad = ids.eq(PAD)

    # Compute the initial token, segment, and positional embeddings.
    # This is the embedding layer x0.
    x = embed_inputs(params, ids, segs)

    # Store the embedding output as the first layer representation.
    hidden_states = [x]

    # Apply each Transformer encoder block sequentially.
    for block in params["blocks"]:
        x = encoder_block(
            block,
            x,
            pad,
            params["n_heads"]
        )

        # Store the output of every encoder layer.
        hidden_states.append(x)

    # Return [x0, x1, ..., xL].
    return hidden_states


def encoder_parameters(params):
    # Begin with the three embedding matrices.
    parameters = [
        params["tok"],
        params["seg"],
        params["pos"]
    ]

    # Collect parameters from every encoder block
    # in the exact order specified by Deep-ML.
    block_keys = (
        "qkv",
        "out",
        "f1",
        "b1",
        "f2",
        "b2",
        "g1",
        "m1",
        "g2",
        "m2"
    )

    for block in params["blocks"]:
        for key in block_keys:
            parameters.append(block[key])

    # Return the flat list of trainable encoder tensors.
    # NSP weights are intentionally excluded.
    return parameters

# Step 12 - mask_tokens
def mask_tokens(ids, rng, vocab_size, k=5, rate=0.15):
    # Convert inputs to a NumPy array without modifying the original.
    ids = np.asarray(ids)

    B, T = ids.shape

    # Create a copy that will contain the corrupted input sequences.
    masked_ids = ids.copy()

    # Allocate fixed-size arrays for masked positions and targets.
    # Unused slots are initialized to zero.
    pos = np.zeros((B, k), dtype=np.int32)
    tgt = np.zeros((B, k), dtype=np.int32)

    # Weights identify which masking slots contain valid targets.
    w = np.zeros((B, k), dtype=np.float32)

    # Process each sequence independently.
    for i in range(B):

        # Only ordinary vocabulary tokens can be selected.
        # Special tokens have indices 0 through 4.
        candidates = np.flatnonzero(ids[i] >= 5)

        if len(candidates) == 0 or k == 0 or rate <= 0:
            continue

        # Determine how many positions to select.
        # Select at least one when candidates are available,
        # but never exceed the fixed number of masking slots.
        n = min(
            k,
            max(1, int(round(len(candidates) * rate)))
        )

        # Randomly select positions without replacement.
        chosen = rng.choice(
            candidates,
            size=n,
            replace=False
        )

        # Sort positions to keep the masking slots ordered.
        chosen.sort()

        # Save the original token positions and targets.
        pos[i, :n] = chosen
        tgt[i, :n] = ids[i, chosen]

        # Mark the selected positions as valid.
        w[i, :n] = 1.0

        # Generate random values for the 80/10/10 strategy.
        draws = rng.random(n)

        # 80%: Replace the selected token with [MASK].
        to_mask = draws < 0.80
        masked_ids[i, chosen[to_mask]] = MASK

        # 10%: Replace the selected token with a random
        # ordinary vocabulary token.
        to_random = (draws >= 0.80) & (draws < 0.90)

        num_random = int(to_random.sum())

        if num_random > 0:
            masked_ids[i, chosen[to_random]] = rng.integers(
                5,
                vocab_size,
                size=num_random
            )

        # Remaining 10%: Leave the original token unchanged.

    return masked_ids, pos, tgt, w

# Step 13 - mlm_loss
def mlm_logits(params, h, pos):
    # Get the batch size, sequence length, and model dimension.
    B, T, d = h.shape

    # Convert the selected positions into one-hot representations.
    # Shape: (B, k, T)
    position_mask = pos.one_hot(T).cast(h.dtype)

    # Select the hidden representations at the masked positions.
    # Shape: (B, k, d)
    selected_hidden = position_mask @ h

    # Use the transpose of the token embedding matrix as the
    # MLM output projection matrix (tied embeddings).
    # Shape: (B, k, vocab_size)
    logits = selected_hidden @ params["tok"].permute(1, 0)

    return logits


def mlm_loss(params, h, pos, tgt, w):
    # Compute vocabulary logits for the selected positions.
    # Shape: (B, k, vocab_size)
    logits = mlm_logits(params, h, pos)

    # Calculate log probabilities using softmax over the vocabulary.
    log_probs = logits.log_softmax(axis=-1)

    # Convert target token IDs into one-hot representations.
    # Shape: (B, k, vocab_size)
    target_one_hot = tgt.one_hot(
        params["vocab_size"]
    ).cast(log_probs.dtype)

    # Calculate the negative log-likelihood for each selected slot.
    # Shape: (B, k)
    nll = -(log_probs * target_one_hot).sum(axis=-1)

    # Convert weights to the same floating-point dtype as the loss.
    weights = w.cast(nll.dtype)

    # Compute the weighted mean negative log-likelihood.
    numerator = (nll * weights).sum()

    # Avoid division by zero when there are no valid masking slots.
    denominator = weights.sum().maximum(1.0)

    return numerator / denominator

# Step 14 - sample_nsp_batch
def sample_nsp_batch(enc, rng, batch, max_len=32):
    # Validate that the corpus contains enough stories and sentences
    # to generate both positive and negative NSP examples.
    if batch > 0:
        if not enc:
            raise ValueError("enc must contain at least one story.")

        if not any(len(story) >= 2 for story in enc):
            raise ValueError("At least one story must contain two sentences.")

        if sum(len(story) > 0 for story in enc) < 2:
            raise ValueError("At least two non-empty stories are required.")

    # Allocate output arrays.
    ids_batch = np.zeros((batch, max_len), dtype=np.int32)
    segs_batch = np.zeros((batch, max_len), dtype=np.int32)
    labels = np.zeros(batch, dtype=np.int32)

    # Generate each NSP example independently.
    for i in range(batch):

        while True:
            # Step 1: Randomly select a story.
            s = int(rng.integers(len(enc)))
            story = enc[s]

            # The selected story must contain at least two sentences.
            if len(story) < 2:
                continue

            # Step 2: Randomly select sentence A.
            j = int(rng.integers(len(story) - 1))

            A = story[j][0]

            # Step 3: With probability 0.5, select the actual
            # next sentence from the same story.
            if rng.random() < 0.5:
                B = story[j + 1][0]
                y = 1

            else:
                # Step 4: Otherwise, select a sentence from
                # a different story to create a negative example.
                o = int(rng.integers(len(enc)))

                # Reject the selection if it is the same story
                # or if the selected story contains no sentences.
                if o == s or len(enc[o]) == 0:
                    continue

                other_story = enc[o]

                # Select a random sentence from the other story.
                other_idx = int(rng.integers(len(other_story)))

                B = other_story[other_idx][0]
                y = 0

            # A valid sentence pair has been generated.
            break

        # Step 5: Pack the sentence pair into BERT input format.
        ids, segs = pack_pair(A, B, max_len=max_len)

        # Store the packed inputs and corresponding NSP label.
        ids_batch[i] = ids
        segs_batch[i] = segs
        labels[i] = y

    return ids_batch, segs_batch, labels

# Step 15 - nsp_loss
def nsp_logits(params, h):
    # Extract the [CLS] representation (position 0) from each sequence.
    # Shape: (B, d_model)
    cls_output = h[:, 0, :]

    # Apply the NSP linear classification layer.
    # Shape: (B, 2)
    logits = cls_output @ params["nsp_w"] + params["nsp_b"]

    return logits


def nsp_loss(params, h, y):
    # Compute NSP classification logits.
    logits = nsp_logits(params, h)

    # Calculate log probabilities over the two NSP classes.
    log_probs = logits.log_softmax(axis=-1)

    # Convert integer labels into one-hot representations.
    # Shape: (B, 2)
    targets = y.one_hot(2).cast(logits.dtype)

    # Calculate negative log-likelihood for each example.
    # Shape: (B,)
    losses = -(log_probs * targets).sum(axis=-1)

    # Return the mean cross-entropy loss across the batch.
    return losses.mean()

# Step 16 - Adam
class Adam:
    def __init__(self, params, b1=0.9, b2=0.999, eps=1e-8):
        # Store the trainable parameters.
        self.params = list(params)

        # Store Adam hyperparameters.
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        # Initialize first and second moment estimates.
        # Each state tensor has the same shape as its parameter.
        self.m = [
            Tensor.zeros(*p.shape).contiguous().realize()
            for p in self.params
        ]

        self.v = [
            Tensor.zeros(*p.shape).contiguous().realize()
            for p in self.params
        ]

        # Initialize the optimization step counter.
        self.t = Tensor([0.0]).contiguous().realize()

    def reset(self):
        # Reset the optimization step counter to zero.
        self.t.assign(self.t * 0).realize()

        # Reset all first and second moment estimates.
        for m, v in zip(self.m, self.v):
            m.assign(m * 0).realize()
            v.assign(v * 0).realize()

    def step(self, grads, lrs):
        # Increment the optimization step counter.
        self.t.assign(self.t + 1.0)

        # Update every parameter using its corresponding gradient
        # and learning rate.
        for p, g, lr, m, v in zip(
            self.params, grads, lrs, self.m, self.v
        ):

            # Update the first moment estimate.
            m.assign(
                self.b1 * m + (1.0 - self.b1) * g
            )

            # Update the second moment estimate.
            v.assign(
                self.b2 * v + (1.0 - self.b2) * g * g
            )

            # Calculate bias-corrected first moment.
            m_hat = m / (1.0 - self.b1 ** self.t)

            # Calculate bias-corrected second moment.
            v_hat = v / (1.0 - self.b2 ** self.t)

            # Update the parameter using the Adam rule.
            p.assign(
                p - lr * m_hat / (v_hat.sqrt() + self.eps)
            )

        # Return all optimizer states and parameters.
        # Do not realize here; the caller handles realization.
        return [
            self.t,
            *self.params,
            *self.m,
            *self.v
        ]

# Step 17 - pretrain
from tinygrad import TinyJit, dtypes


def lr_at(t, steps, lr, warmup=100):
    # Linear learning-rate warmup.
    if t < warmup:
        return lr * (t + 1) / warmup

    # Linearly decay the learning rate to 5% of its initial value.
    decay = 1.0 - (t - warmup) / max(1, steps - warmup)

    return lr * (0.05 + 0.95 * decay)


def pretrain(params, enc, steps=2000, batch=32, lr=2e-3,
             k=8, rate=0.25, warmup=100, seed=0, log_every=250):

    if log_every <= 0:
        raise ValueError("log_every must be positive")

    # ---------------------------------------------------------
    # 1. Initialize the random generator and Adam optimizer.
    # ---------------------------------------------------------

    rng = np.random.default_rng(seed)

    # Collect all encoder parameters and NSP classifier parameters.
    train_params = (
        encoder_parameters(params)
        + [params["nsp_w"], params["nsp_b"]]
    )

    opt = Adam(train_params)

    history = []

    # ---------------------------------------------------------
    # 2. Define the training step.
    # ---------------------------------------------------------

    def train_step(x, segs, pos, tgt, w, y, lr_tensor):

        # Run the bidirectional encoder and obtain its final layer.
        h = encode(params, x, segs)[-1]

        # Compute the masked language modelling loss.
        mlm = mlm_loss(params, h, pos, tgt, w)

        # Compute the next-sentence prediction loss.
        nsp = nsp_loss(params, h, y)

        # Combine both objectives.
        total_loss = mlm + nsp

        # Calculate gradients for all trainable parameters.
        grads = total_loss.gradient(*train_params)

        # Apply the Adam optimizer using the supplied learning rate.
        # The same learning rate is used for every parameter.
        updates = opt.step(
            grads,
            [lr_tensor] * len(train_params)
        )

        # Realize both losses and all optimizer updates together.
        Tensor.realize(mlm, nsp, *updates)

        return mlm, nsp

    # Compile the training step once using TinyJit.
    train_step = TinyJit(train_step)

    # ---------------------------------------------------------
    # 3. Run the self-supervised training loop.
    # ---------------------------------------------------------

    for t in range(steps):

        # Sample next-sentence prediction examples.
        ids, segs, y = sample_nsp_batch(
            enc,
            rng,
            batch,
            params["max_len"]
        )

        # Apply masked language modelling corruption.
        x, pos, tgt, w = mask_tokens(
            ids,
            rng,
            params["vocab_size"],
            k=k,
            rate=rate
        )

        # Convert NumPy arrays into tinygrad tensors.
        x_tensor = Tensor(x, dtype=dtypes.int32)
        segs_tensor = Tensor(segs, dtype=dtypes.int32)
        pos_tensor = Tensor(pos, dtype=dtypes.int32)
        tgt_tensor = Tensor(tgt, dtype=dtypes.int32)
        w_tensor = Tensor(w, dtype=dtypes.float32)
        y_tensor = Tensor(y, dtype=dtypes.int32)

        # Calculate the scheduled learning rate.
        current_lr = lr_at(
            t,
            steps,
            lr,
            warmup
        )

        # TinyJit receives the learning rate as a one-element tensor.
        lr_tensor = Tensor(
            [current_lr],
            dtype=dtypes.float32
        )

        # Perform one optimization step.
        mlm_value, nsp_value = train_step(
            x_tensor,
            segs_tensor,
            pos_tensor,
            tgt_tensor,
            w_tensor,
            y_tensor,
            lr_tensor
        )

        # -----------------------------------------------------
        # 4. Record training history at the requested intervals.
        # -----------------------------------------------------

        if t % log_every == 0 or t == steps - 1:

            mlm_score = float(mlm_value.numpy().item())
            nsp_score = float(nsp_value.numpy().item())

            history.append({
                "t": t,
                "mlm": mlm_score,
                "nsp": nsp_score
            })

    # Return the training history.
    # The parameters have already been updated in place.
    return history

# Step 18 - mlm_metrics
def encode_all(params, ids, segs, bs=256, layers=False):
    # Convert inputs to NumPy arrays without modifying the originals.
    ids = np.asarray(ids, dtype=np.int32)
    segs = np.asarray(segs, dtype=np.int32)

    # Validate the input dimensions.
    if ids.ndim != 2 or segs.ndim != 2:
        raise ValueError("ids and segs must be 2D arrays.")

    if ids.shape != segs.shape:
        raise ValueError("ids and segs must have identical shapes.")

    if bs <= 0:
        raise ValueError("bs must be a positive integer.")

    # Get the number of input sequences.
    N, T = ids.shape

    # Handle an empty input batch.
    if N == 0:
        empty = np.empty(
            (0, T, params["d_model"]),
            dtype=np.float32
        )

        if layers:
            return [
                empty.copy()
                for _ in range(params["n_layers"] + 1)
            ]

        return empty

    # Create storage for the output of each encoder layer.
    layer_outputs = [
        []
        for _ in range(params["n_layers"] + 1)
    ]

    # Process the input sequences in batches.
    for start in range(0, N, bs):
        end = min(start + bs, N)

        # Convert the current NumPy batch into tinygrad tensors.
        ids_batch = Tensor(ids[start:end])
        segs_batch = Tensor(segs[start:end])

        # Run the complete encoder.
        # Returns [x0, x1, ..., xL].
        hidden_states = encode(
            params,
            ids_batch,
            segs_batch
        )

        # Convert every layer's output into a NumPy array.
        # Calling numpy() realizes the computation.
        for layer_idx, hidden in enumerate(hidden_states):
            layer_outputs[layer_idx].append(hidden.numpy())

    # Concatenate the batches for each encoder layer.
    all_layers = [
        np.concatenate(outputs, axis=0)
        for outputs in layer_outputs
    ]

    # Return all layers when requested.
    if layers:
        return all_layers

    # Otherwise, return only the final encoder layer.
    return all_layers[-1]


def mlm_metrics(params, enc, n=512, seed=123, k=5, rate=0.15):
    # Initialize a reproducible random number generator.
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Generate an evaluation batch.
    # ---------------------------------------------------------

    ids, segs, y = sample_nsp_batch(
        enc,
        rng,
        n,
        params["max_len"]
    )

    # Apply masked language modelling corruption.
    x, pos, tgt, w = mask_tokens(
        ids,
        rng,
        params["vocab_size"],
        k=k,
        rate=rate
    )

    # ---------------------------------------------------------
    # 2. Run the encoder.
    # ---------------------------------------------------------

    # Obtain the final hidden representations as a NumPy array.
    h_array = encode_all(
        params,
        x,
        segs,
        layers=False
    )

    # Convert the final hidden representations into a Tensor.
    h = Tensor(h_array)

    # ---------------------------------------------------------
    # 3. Calculate masked language model logits.
    # ---------------------------------------------------------

    logits = mlm_logits(
        params,
        h,
        Tensor(pos)
    ).numpy()

    # Identify the valid masking positions.
    valid = w > 0

    # Extract only the actual prediction targets.
    targets = tgt[valid]

    if len(targets) > 0:

        # Top-1 accuracy.
        predictions = logits.argmax(axis=-1)

        top1 = float(
            np.mean(predictions[valid] == targets)
        )

        # Top-5 accuracy.
        vocab_size = params["vocab_size"]
        top_k = min(5, vocab_size)

        top_indices = np.argsort(
            logits,
            axis=-1
        )[..., -top_k:]

        top5_correct = np.any(
            top_indices == tgt[..., None],
            axis=-1
        )

        top5 = float(
            np.mean(top5_correct[valid])
        )

        # Most-common-target baseline.
        # Always predict the most frequent target in this batch.
        counts = np.bincount(
            targets,
            minlength=vocab_size
        )

        most_common_id = int(np.argmax(counts))

        most_common = float(
            np.mean(targets == most_common_id)
        )

    else:
        # Handle batches without any valid masked positions.
        top1 = 0.0
        top5 = 0.0
        most_common = 0.0

    # ---------------------------------------------------------
    # 4. Calculate next-sentence prediction accuracy.
    # ---------------------------------------------------------

    nsp_scores = nsp_logits(
        params,
        h
    ).numpy()

    nsp_predictions = nsp_scores.argmax(axis=-1)

    if len(y) > 0:
        nsp = float(
            np.mean(nsp_predictions == y)
        )
    else:
        nsp = 0.0

    # ---------------------------------------------------------
    # 5. Return all evaluation metrics as Python floats.
    # ---------------------------------------------------------

    return {
        "top1": top1,
        "top5": top5,
        "most_common": most_common,
        "nsp": nsp
    }

