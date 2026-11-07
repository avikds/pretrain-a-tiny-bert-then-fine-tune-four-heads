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
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

_DATA = {
    "stories": [],
    "loaded": False,
}


def load_stories(start=0, stop=1000):
    """
    Load TinyStories directly from its public Hugging Face
    Parquet shard, without using the datasets library.

    Stories are cached so subsequent calls do not download
    or read the dataset again.
    """

    if start < 0 or stop < start:
        raise ValueError("Invalid story slice.")

    # Load the dataset only once.
    if not _DATA["loaded"]:

        filename = hf_hub_download(
            repo_id="roneneldan/TinyStories",
            repo_type="dataset",
            filename="data/train-00000-of-00004-2d5a1467fff1081b.parquet"
        )

        parquet_file = pq.ParquetFile(filename)

        stories = []

        # Read only as many rows as necessary.
        for batch in parquet_file.iter_batches(
            columns=["text"],
            batch_size=1024
        ):
            stories.extend(batch.column("text").to_pylist())

            # The project requests at most 12000 stories.
            if len(stories) >= max(stop, 12000):
                break

        _DATA["stories"] = stories
        _DATA["loaded"] = True

    # Convert each story into the project's sentence format.
    return [
        split_sentences(story)
        for story in _DATA["stories"][start:stop]
    ]

# Step 3 - entity_lists
import numpy as np
import random
import names as name_data
from collections import Counter


CAST = (
    "lily", "timmy", "ben", "tom", "max", "tim", "sam", "jack",
    "lucy", "anna", "mia", "sarah", "jane", "bob", "john",
    "sue", "benny", "sally", "joe", "billy", "sara", "tommy",
    "emma", "amy", "sammy", "molly", "jill", "lila", "jake",
    "jenny", "emily", "jimmy", "jen", "mary", "lola", "peter",
    "johnny", "jim", "pete", "ella", "kim", "mandy", "bella",
    "leo", "alex", "ted", "dave", "lisa", "mike", "henry",
    "jerry", "susie", "ann", "andy", "charlie", "tony", "suzy",
    "clara", "alice", "fred", "lilly", "steve", "maria",
    "bobby", "harry", "james", "joey", "millie", "sophie",
    "ellie", "thomas", "toby", "matt", "george", "polly",
    "zoe", "paul", "julie", "suzie", "chloe", "kate",
    "jeff", "chris"
)

SURNAMES = (
    "green", "brown", "black", "white", "gray", "fox", "bird",
    "wolf", "bear", "hill", "wood", "stone", "lake", "king",
    "baker", "cook", "hunter", "farmer", "bell", "rose",
    "snow", "day", "long", "young", "little", "pond", "field"
)

PLACE_A = (
    "new", "old", "north", "south", "east", "west", "green",
    "oak", "silver", "red", "blue", "big", "little", "long",
    "high", "sun", "fair", "rock", "clear", "white"
)

PLACE_B = (
    "york", "town", "hill", "lake", "city", "valley", "river",
    "bay", "falls", "port", "field", "wood", "land", "beach",
    "island", "harbor", "ridge", "creek", "forest", "spring"
)


def entity_lists(k=100, seed=0):

    # Load the first 3000 TinyStories.
    stories = load_stories(0, 3000)

    # Count token frequencies.
    token_counts = Counter(
        token
        for story in stories
        for sentence in story
        for token in sentence
    )

    cast_set = set(CAST)

    # Identify common words.
    common = {
        word
        for word, count in token_counts.items()
        if count >= 3 and word not in cast_set
    }

    # Generate a reproducible local first-name registry.
    # Preserve the caller's global random state.
    old_state = random.getstate()

    try:
        random.seed(1729)

        registry_names = []
        seen = set()

        for _ in range(50000):
            name = name_data.get_first_name().lower()

            if name.isalpha() and name not in seen:
                seen.add(name)
                registry_names.append(name)

    finally:
        random.setstate(old_state)

    # Exclude common words, surnames and place components.
    excluded = common | set(SURNAMES) | set(PLACE_A) | set(PLACE_B)

    names = []

    for name in registry_names:

        if len(names) >= k:
            break

        if len(name) > 2 and name not in excluded:
            names.append(name)

    # Generate all place combinations in nested order.
    places = [
        (a, b)
        for a in PLACE_A
        for b in PLACE_B
    ]

    # Reproducible train/held-out splits.
    rng = np.random.default_rng(seed)

    name_idx = rng.permutation(len(names))

    train_names = [names[i] for i in name_idx[::2]]
    heldout_names = [names[i] for i in name_idx[1::2]]

    place_idx = rng.permutation(len(places))

    train_places = [places[i] for i in place_idx[::2]]
    heldout_places = [places[i] for i in place_idx[1::2]]

    return {
        "train": (
            train_names,
            train_places
        ),

        "heldout": (
            heldout_names,
            heldout_places
        ),

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

# Step 19 - cloze_topk
def cloze_topk(params, vocab, text, topk=5):
    # Split around special markers
    parts = re.split(
        r'(\[MASK\]|\[SEP\])',
        text,
        flags=re.IGNORECASE
    )

    a, b = [], []
    current = a
    has_sep = False
    mask_a, mask_b = [], []

    for part in parts:
        upper = part.upper()

        if upper == "[MASK]":
            current.append(MASK)
            if current is a:
                mask_a.append(len(a) - 1)
            else:
                mask_b.append(len(b) - 1)

        elif upper == "[SEP]":
            current = b
            has_sep = True

        elif part:
            tokens = TOKEN.findall(part.lower())
            current.extend(vocab.get(w, UNK) for w in tokens)

    # Pack inputs
    ids, segs = pack_pair(
        a, b if has_sep else None,
        max_len=params["max_len"]
    )

    # Determine positions after truncation
    positions = []

    if has_sep:
        na = len(a)
        nb = len(b)

        # Account for truncation performed by pack_pair
        while 3 + na + nb > params["max_len"]:
            if na >= nb:
                na -= 1
            else:
                nb -= 1

        positions.extend(
            1 + p for p in mask_a if p < na
        )

        positions.extend(
            na + 3 + p for p in mask_b if p < nb
        )
    else:
        positions.extend(
            1 + p for p in mask_a
            if p < params["max_len"] - 2
        )

    if not positions:
        return []

    # Encode
    h = encode_all(
        params,
        ids[None, :],
        segs[None, :]
    )[0]

    # Tied embeddings
    E = params["tok"].numpy()
    words = sorted(vocab, key=vocab.get)

    output = []

    for pos in positions:
        logits = h[pos] @ E.T
        best = np.argsort(-logits, kind="stable")[:topk]
        output.append([words[i] for i in best])

    return output


def future_context_gain(params, enc, n=400, seed=7):
    rng = np.random.default_rng(seed)

    E = params["tok"].numpy()
    max_len = params["max_len"]

    both_correct = 0
    left_correct = 0
    count = 0

    while count < n:
        s = int(rng.integers(len(enc)))
        story = enc[s]

        if len(story) == 0:
            continue

        j = int(rng.integers(len(story)))

        if story[j] is None:
            continue

        t = np.asarray(story[j][0], dtype=np.int32)

        if not (6 <= len(t) <= max_len - 2):
            continue

        p = int(rng.integers(2, len(t) - 1))

        if t[p] < 5:
            continue

        target = int(t[p])

        ids, segs = pack_pair(t, max_len=max_len)
        pos = p + 1

        x = ids.copy()
        x[pos] = MASK

        # Full context
        h = encode_all(
            params, x[None, :], segs[None, :]
        )[0]

        pred_both = int(np.argmax(h[pos] @ E.T))

        # Left context only
        x_left = x.copy()
        x_left[pos + 1:] = PAD

        h_left = encode_all(
            params, x_left[None, :], segs[None, :]
        )[0]

        pred_left = int(np.argmax(h_left[pos] @ E.T))

        both_correct += int(pred_both == target)
        left_correct += int(pred_left == target)

        count += 1

    return (
        both_correct / count,
        left_correct / count
    )

# Step 20 - scalar_mix_probe
def scalar_mix_probe(features, labels, n_classes,
                     steps=300, lr=0.02, seed=0, batch=256):

    # Convert inputs to NumPy arrays with the required data types.
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    # Get the number of tokens, layers, and feature dimensions.
    N, M, d = features.shape

    if N == 0:
        raise ValueError("features must contain at least one example.")

    if len(labels) != N:
        raise ValueError("features and labels must have equal lengths.")

    if n_classes <= 0 or batch <= 0 or steps < 0:
        raise ValueError("Invalid n_classes, batch, or steps.")

    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("Labels must be valid class indices.")

    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Initialize the scalar-mix and classifier parameters.
    # ---------------------------------------------------------

    # One scalar mixing parameter per encoder layer.
    a = _weight(np.zeros(M, dtype=np.float32))

    # Linear classifier weights.
    W = _weight(
        rng.normal(0.0, 0.01, size=(d, n_classes))
    )

    # Linear classifier bias.
    b = _weight(np.zeros(n_classes, dtype=np.float32))

    # Initialize Adam for all three trainable tensors.
    opt = Adam([a, W, b])

    # Convert the frozen features and labels into tinygrad tensors.
    features_t = Tensor(features, dtype=dtypes.float32)
    labels_t = Tensor(labels, dtype=dtypes.int32)

    # ---------------------------------------------------------
    # 2. Define the TinyJit training step.
    # ---------------------------------------------------------

    def train_step(idx, lr_tensor):

        # Select the current minibatch.
        batch_features = features_t[idx]
        batch_labels = labels_t[idx]

        # Calculate normalized scalar-mix weights.
        # Shape: (M,)
        mix_weights = a.softmax(axis=0)

        # Combine the representations from all layers.
        # Shape: (batch_size, d)
        mixed_features = (
            batch_features * mix_weights.reshape(1, M, 1)
        ).sum(axis=1)

        # Apply the linear classification layer.
        # Shape: (batch_size, n_classes)
        logits = mixed_features @ W + b

        # Calculate log probabilities.
        log_probs = logits.log_softmax(axis=-1)

        # Convert labels into one-hot representations.
        targets = batch_labels.one_hot(n_classes).cast(logits.dtype)

        # Calculate mean softmax cross-entropy.
        loss = -(
            log_probs * targets
        ).sum(axis=-1).mean()

        # Calculate gradients for scalar-mix and classifier parameters.
        grads = loss.gradient(a, W, b)

        # Update parameters with Adam.
        updates = opt.step(
            grads,
            [lr_tensor, lr_tensor, lr_tensor]
        )

        # Realize the loss and optimizer updates.
        Tensor.realize(loss, *updates)

        return loss

    # Compile the training step using TinyJit.
    train_step = TinyJit(train_step)

    # ---------------------------------------------------------
    # 3. Train the scalar-mix probe.
    # ---------------------------------------------------------

    for t in range(steps):

        # Draw a random minibatch without replacement.
        batch_size = min(batch, N)

        idx = rng.integers(
            0, N, size=batch_size
        )

        # Calculate the scheduled learning rate.
        current_lr = lr * (1.0 - t / max(1, steps)) + 1e-4

        # Pass indices and learning rate as tensors.
        idx_t = Tensor(idx, dtype=dtypes.int32)

        lr_tensor = Tensor(
            [current_lr],
            dtype=dtypes.float32
        )

        # Perform one training update.
        train_step(idx_t, lr_tensor)

    # ---------------------------------------------------------
    # 4. Return the trained probe parameters.
    # ---------------------------------------------------------

    return {
        "weights": a.softmax(axis=0).numpy(),
        "W": W.numpy(),
        "b": b.numpy()
    }


def probe_accuracy(features, probe, labels):

    # Convert inputs into NumPy arrays.
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    # Retrieve the learned scalar-mix weights and classifier.
    weights = np.asarray(probe["weights"])
    W = np.asarray(probe["W"])
    b = np.asarray(probe["b"])

    # Combine the representations from all layers.
    # Shape: (N, d)
    mixed_features = np.einsum(
        "m,nmd->nd",
        weights,
        features
    )

    # Calculate class logits.
    logits = mixed_features @ W + b

    # Select the highest-scoring class.
    predictions = logits.argmax(axis=-1)

    # Return classification accuracy as a Python float.
    return float(np.mean(predictions == labels))

# Step 21 - layer_probes
FUNCTION_WORDS = {
    "det": ("the", "a", "an", "this", "that", "these", "those", "his", "her", "their", "my", "your", "our", "its"),
    "pron": ("he", "she", "it", "they", "i", "you", "we", "him", "them", "me", "us"),
    "prep": ("to", "at", "in", "on", "of", "for", "with", "from", "by", "near", "under", "over", "into"),
    "conj": ("and", "but", "or", "so", "because", "when", "if", "then"),
    "aux": ("was", "were", "is", "are", "am", "had", "have", "has", "did", "do", "does", "will", "can", "could", "would"),
    "punct": (".", ",", "!", "?"),
}


def token_class_set(enc, vocab, n_seq=400, seed=0, max_len=32):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    if n_seq < 0:
        raise ValueError("n_seq must be non-negative.")

    if n_seq == 0:
        return (
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0, max_len), dtype=np.int32)
        )

    if not enc:
        raise ValueError("enc must contain at least one story.")

    # Map vocabulary IDs to their corresponding function-word classes.
    # Class 0 is reserved for all other tokens.
    id_to_class = {}

    for class_idx, words in enumerate(FUNCTION_WORDS.values(), start=1):
        for word in words:
            if word in vocab:
                id_to_class[vocab[word]] = class_idx

    # Verify that at least one eligible sentence exists.
    has_valid_sentence = any(
        4 <= len(sentence[0]) <= max_len - 2
        for story in enc
        for sentence in story
    )

    if not has_valid_sentence:
        raise ValueError("No eligible sentences found in enc.")

    # Allocate the output arrays.
    ids_batch = np.zeros((n_seq, max_len), dtype=np.int32)
    segs_batch = np.zeros((n_seq, max_len), dtype=np.int32)
    labels = np.zeros((n_seq, max_len), dtype=np.int32)

    # Draw sentences independently until n_seq valid examples are collected.
    for i in range(n_seq):

        while True:
            # Select a random story.
            s = int(rng.integers(len(enc)))
            story = enc[s]

            if len(story) == 0:
                continue

            # Select a random sentence from that story.
            j = int(rng.integers(len(story)))
            sentence = np.asarray(story[j][0], dtype=np.int32)

            # Accept only sentences with the required length.
            if 4 <= len(sentence) <= max_len - 2:
                break

        # Pack the sentence as a single BERT input.
        ids, segs = pack_pair(sentence, max_len=max_len)

        ids_batch[i] = ids
        segs_batch[i] = segs

        # Assign function-word classes to the original token positions.
        # Position 0 is [CLS], so sentence positions begin at 1.
        for p, token_id in enumerate(sentence):
            labels[i, p + 1] = id_to_class.get(int(token_id), 0)

    return ids_batch, segs_batch, labels


def sense_set(stories, vocab, per_word=30, seed=0, max_len=32):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    if per_word < 0:
        raise ValueError("per_word must be non-negative.")

    # Store the selected sentence occurrences and their labels.
    selected = []

    # Process each surname independently.
    for surname in SURNAMES:

        # Only consider surnames present in the vocabulary.
        if surname not in vocab:
            continue

        positives = []
        negatives = []

        # Find every occurrence of the surname in the tagged corpus.
        for story in stories:
            for tokens, tags in story:

                # Only consider sentences within the required length.
                if not (4 <= len(tokens) <= max_len - 2):
                    continue

                # Find every occurrence, including repeated occurrences
                # within the same sentence.
                for p, (word, tag) in enumerate(zip(tokens, tags)):

                    if word != surname:
                        continue

                    occurrence = (tokens, p)

                    # Tag 2: surname is part of a person's name.
                    if tag == 2:
                        positives.append(occurrence)

                    # Tag 0: surname is an ordinary word.
                    elif tag == 0:
                        negatives.append(occurrence)

        # Shuffle positive and negative examples independently.
        rng.shuffle(positives)
        rng.shuffle(negatives)

        # Keep equal numbers of positive and negative examples.
        m = min(
            per_word,
            len(positives),
            len(negatives)
        )

        if m == 0:
            continue

        selected.extend(
            (occurrence, 1)
            for occurrence in positives[:m]
        )

        selected.extend(
            (occurrence, 0)
            for occurrence in negatives[:m]
        )

    # Allocate lists for the packed examples.
    all_ids = []
    all_segs = []
    all_cols = []
    all_labels = []

    # Pack each selected occurrence into a BERT input.
    for (tokens, p), label in selected:

        # Convert the sentence into vocabulary IDs.
        sentence_ids = [
            vocab.get(word, UNK)
            for word in tokens
        ]

        # Pack the sentence independently.
        ids, segs = pack_pair(
            sentence_ids,
            max_len=max_len
        )

        # Account for the [CLS] token at position 0.
        packed_col = p + 1

        all_ids.append(ids)
        all_segs.append(segs)
        all_cols.append(packed_col)
        all_labels.append(label)

    # Convert the output collections into NumPy arrays.
    if len(all_ids) == 0:
        return (
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32)
        )

    return (
        np.asarray(all_ids, dtype=np.int32),
        np.asarray(all_segs, dtype=np.int32),
        np.asarray(all_cols, dtype=np.int32),
        np.asarray(all_labels, dtype=np.int32)
    )


def layer_probes(params, vocab, train_stories, test_stories,
                 seed=0, steps=300):

    # Number of encoder layers plus the embedding layer.
    n_layers = params["n_layers"] + 1
    d = params["d_model"]
    max_len = params["max_len"]

    # ---------------------------------------------------------
    # Helper: extract hidden representations from every layer.
    # ---------------------------------------------------------

    def extract_layers(ids, segs):
        # encode_all returns a list of L+1 arrays,
        # each having shape (N, T, d).
        hidden_states = encode_all(
            params,
            ids,
            segs,
            bs=256,
            layers=True
        )

        # Stack the layers along axis 2.
        # Result: (N, T, L+1, d)
        return np.stack(hidden_states, axis=2)

    # ---------------------------------------------------------
    # Helper: train and evaluate scalar-mix and single-layer probes.
    # ---------------------------------------------------------

    def evaluate_task(train_features, train_labels,
                      test_features, test_labels, n_classes):

        if len(train_labels) == 0 or len(test_labels) == 0:
            raise ValueError("Probe datasets must not be empty.")

        # Train one probe using all encoder layers.
        mix_probe = scalar_mix_probe(
            train_features,
            train_labels,
            n_classes,
            steps=steps,
            seed=seed
        )

        # Evaluate the learned scalar mix.
        mix_acc = probe_accuracy(
            test_features,
            mix_probe,
            test_labels
        )

        # Train a separate linear probe for every individual layer.
        layer_acc = []

        for layer_idx in range(n_layers):

            # Select only the current layer.
            # Shape: (N, 1, d)
            train_single = train_features[
                :, layer_idx:layer_idx + 1, :
            ]

            test_single = test_features[
                :, layer_idx:layer_idx + 1, :
            ]

            # Train the single-layer probe using half the steps.
            single_probe = scalar_mix_probe(
                train_single,
                train_labels,
                n_classes,
                steps=steps // 2,
                seed=seed
            )

            # Evaluate on the corresponding test-layer features.
            accuracy = probe_accuracy(
                test_single,
                single_probe,
                test_labels
            )

            layer_acc.append(float(accuracy))

        return {
            "weights": [
                float(w) for w in mix_probe["weights"]
            ],
            "mix_acc": float(mix_acc),
            "layer_acc": layer_acc
        }

    # =========================================================
    # TASK 1: SYNTACTIC PROBE — CLOSED-CLASS WORD TAGGING
    # =========================================================

    # Encode the tagged training and testing stories.
    train_enc = encode_stories(train_stories, vocab)
    test_enc = encode_stories(test_stories, vocab)

    # Generate syntactic training examples.
    tr_ids, tr_segs, tr_labels = token_class_set(
        train_enc,
        vocab,
        n_seq=400,
        seed=seed,
        max_len=max_len
    )

    # Generate syntactic testing examples using a separate seed.
    te_ids, te_segs, te_labels = token_class_set(
        test_enc,
        vocab,
        n_seq=400,
        seed=seed + 1,
        max_len=max_len
    )

    # Extract all layer representations.
    tr_hidden = extract_layers(tr_ids, tr_segs)
    te_hidden = extract_layers(te_ids, te_segs)

    # Keep all ordinary vocabulary positions.
    # Special tokens, padding and UNK are excluded.
    tr_valid = tr_ids >= 5
    te_valid = te_ids >= 5

    # Extract token-level features.
    # Shape: (number of valid tokens, L+1, d)
    syn_train_features = tr_hidden[tr_valid]
    syn_test_features = te_hidden[te_valid]

    # Extract corresponding classification labels.
    syn_train_labels = tr_labels[tr_valid]
    syn_test_labels = te_labels[te_valid]

    # Train and evaluate the seven-class syntactic probe.
    syntactic_result = evaluate_task(
        syn_train_features,
        syn_train_labels,
        syn_test_features,
        syn_test_labels,
        n_classes=len(FUNCTION_WORDS) + 1
    )

    # =========================================================
    # TASK 2: SEMANTIC PROBE — WORD SENSE CLASSIFICATION
    # =========================================================

    # Generate balanced surname-occurrence training examples.
    tr_ids, tr_segs, tr_cols, tr_y = sense_set(
        train_stories,
        vocab,
        per_word=30,
        seed=seed,
        max_len=max_len
    )

    # Generate balanced surname-occurrence testing examples.
    te_ids, te_segs, te_cols, te_y = sense_set(
        test_stories,
        vocab,
        per_word=30,
        seed=seed + 1,
        max_len=max_len
    )

    # Extract hidden representations from every layer.
    tr_hidden = extract_layers(tr_ids, tr_segs)
    te_hidden = extract_layers(te_ids, te_segs)

    # Select the representation at each surname occurrence.
    # Each result has shape: (N, L+1, d).
    tr_rows = np.arange(len(tr_ids))
    te_rows = np.arange(len(te_ids))

    sense_train_features = tr_hidden[
        tr_rows,
        tr_cols
    ]

    sense_test_features = te_hidden[
        te_rows,
        te_cols
    ]

    # Train and evaluate the binary semantic probe.
    sense_result = evaluate_task(
        sense_train_features,
        tr_y,
        sense_test_features,
        te_y,
        n_classes=2
    )

    # ---------------------------------------------------------
    # Return the results for both tasks.
    # ---------------------------------------------------------

    return {
        "syntactic": syntactic_result,
        "sense": sense_result
    }

# Step 22 - MLPHead
class MLPHead:
    def __init__(self, d, n_out, seed=0):
        # Initialize a reproducible random number generator.
        rng = np.random.default_rng(seed)

        # First linear layer: (d, d).
        self.w1 = _weight(
            rng.normal(
                0.0,
                math.sqrt(2.0 / d),
                size=(d, d)
            )
        )

        # Second linear layer: (d, n_out).
        self.w2 = _weight(
            rng.normal(
                0.0,
                0.01,
                size=(d, n_out)
            )
        )

        # Initialize biases to zero.
        self.b1 = _weight(np.zeros(d, dtype=np.float32))
        self.b2 = _weight(np.zeros(n_out, dtype=np.float32))

    def parameters(self):
        # Return all trainable head parameters in the required order.
        return [self.w1, self.b1, self.w2, self.b2]

    def __call__(self, x):
        # First linear layer followed by GELU.
        hidden = (x @ self.w1 + self.b1).gelu()

        # Second linear layer produces the output logits.
        return hidden @ self.w2 + self.b2


def snapshot(tensors):
    # Return independent NumPy copies of all supplied tensors.
    return [
        tensor.numpy().copy()
        for tensor in tensors
    ]


def restore(tensors, arrays):
    # Ensure that the checkpoint contains the correct number of arrays.
    if len(tensors) != len(arrays):
        raise ValueError(
            "The number of tensors and checkpoint arrays must match."
        )

    # Restore values into the existing tensors.
    for tensor, array in zip(tensors, arrays):

        # Convert the checkpoint array to a tinygrad tensor
        # with the same dtype as the destination tensor.
        value = Tensor(
            np.asarray(array, dtype=np.float32),
            dtype=tensor.dtype
        )

        # Assign the restored values and realize the update.
        tensor.assign(value).realize()

# Step 23 - FineTuner
class FineTuner:
    def __init__(self, params, head_params, loss_fn):
        # Store the encoder parameters.
        self.params = params

        # Store the head parameters.
        self.head_params = list(head_params)

        # Store the task-specific loss function.
        self.loss_fn = loss_fn

        # Collect all encoder parameters.
        self.enc_params = encoder_parameters(params)

        # Combine encoder and head parameters.
        self.all_params = self.enc_params + self.head_params

        # Initialize Adam over all trainable parameters.
        self.opt = Adam(self.all_params)

        # -----------------------------------------------------
        # Define the single compiled training step.
        # -----------------------------------------------------

        def train_step(ids, segs, *args):

            # The final two arguments are the encoder and head
            # learning-rate tensors.
            lr_enc_tensor = args[-2]
            lr_head_tensor = args[-1]

            # All preceding extra arguments belong to the batch.
            extra = args[:-2]

            # Run the encoder and obtain every hidden layer.
            layers = encode(
                self.params,
                ids,
                segs
            )

            # Calculate the task-specific loss.
            loss = self.loss_fn(layers, *extra)

            # Calculate gradients for all encoder and head parameters.
            grads = loss.gradient(*self.all_params)

            # Give each encoder parameter its encoder learning rate.
            # Give each head parameter its head learning rate.
            learning_rates = (
                [lr_enc_tensor] * len(self.enc_params)
                + [lr_head_tensor] * len(self.head_params)
            )

            # Apply Adam updates.
            updates = self.opt.step(
                grads,
                learning_rates
            )

            # Realize the loss and all optimizer outputs.
            Tensor.realize(loss, *updates)

            return loss

        # Compile the training step exactly once.
        self._train_step = TinyJit(train_step)

    def run(self, batch_fn, steps, lr_enc, lr_head,
            seed=0, warmup=20):

        if steps <= 0:
            raise ValueError("steps must be positive.")

        # Initialize the random number generator.
        rng = np.random.default_rng(seed)

        # Reset Adam's step counter and moment estimates.
        self.opt.reset()

        last_loss = None

        # -----------------------------------------------------
        # Run the training loop.
        # -----------------------------------------------------

        for t in range(steps):

            # Generate the NumPy training batch.
            # Expected format: (ids, segs, *extra).
            batch = batch_fn(rng)

            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "batch_fn must return (ids, segs, *extra)."
                )

            # Convert all NumPy batch arrays into tinygrad tensors.
            batch_tensors = []

            for array in batch:
                array = np.asarray(array)

                # Use float32 for floating-point batch arrays
                # and int32 for integer batch arrays.
                if array.dtype.kind == "f":
                    array = array.astype(np.float32, copy=False)
                elif array.dtype.kind in "iu":
                    array = array.astype(np.int32, copy=False)

                batch_tensors.append(Tensor(array))

            # -------------------------------------------------
            # Calculate the learning-rate warmup factor.
            # -------------------------------------------------

            f = lr_at(
                t,
                steps,
                1.0,
                warmup
            )

            # Create separate one-element learning-rate tensors.
            lr_enc_tensor = Tensor(
                [lr_enc * f],
                dtype=dtypes.float32
            )

            lr_head_tensor = Tensor(
                [lr_head * f],
                dtype=dtypes.float32
            )

            # -------------------------------------------------
            # Execute the compiled training step.
            # -------------------------------------------------

            last_loss = self._train_step(
                *batch_tensors,
                lr_enc_tensor,
                lr_head_tensor
            )

        # Return the final loss as a Python float.
        return float(last_loss.numpy().item())

# Step 24 - mention_set
TOPICS = {
    "animal": (
        "dog", "cat", "bird", "bear", "rabbit", "bunny",
        "fish", "frog", "duck", "mouse", "lion", "monkey",
        "cow", "pig", "horse", "sheep", "squirrel", "owl",
        "butterfly", "bee", "elephant", "turtle", "deer", "tiger"
    ),

    "food": (
        "cake", "apple", "cookie", "cookies", "soup", "bread",
        "candy", "pie", "banana", "juice", "milk", "carrot",
        "sandwich", "cheese", "egg", "chocolate", "lemon", "pear",
        "rice", "corn", "honey", "meat", "orange", "berries"
    ),

    "thing": (
        "ball", "doll", "car", "truck", "teddy", "kite",
        "blocks", "robot", "train", "boat", "plane", "bike",
        "puzzle", "balloon", "drum", "bus", "rocket", "ship",
        "wagon", "toys", "shoes", "hat", "dress", "book"
    ),
}


def mention_set(stories, n, vocab, seed=0, max_len=32):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Collect sentences separately for each class.
    # 0 = neither, 1 = person, 2 = place, 3 = both.
    class_rows = {c: [] for c in range(4)}

    # Collect eligible sentences from the tagged stories.
    for story in stories:
        for tokens, tags in story:

            # Keep sentences of length 4 through max_len - 2.
            if not (4 <= len(tokens) <= max_len - 2):
                continue

            has_person = 1 in tags
            has_place = 3 in tags

            # Determine the sentence label.
            label = int(has_person) + 2 * int(has_place)

            class_rows[label].append(tokens)

    # Sample up to n // 4 sentences from each class.
    per_class = n // 4
    selected = []

    for c in range(4):
        rows = class_rows[c]

        order = rng.permutation(len(rows))

        for idx in order[:per_class]:
            selected.append((rows[idx], c))

    # Randomly permute all selected examples.
    order = rng.permutation(len(selected))
    selected = [selected[i] for i in order]

    all_ids = []
    all_segs = []
    all_labels = []

    # Encode and pack each sentence independently.
    for tokens, label in selected:
        token_ids = [
            vocab.get(word, UNK)
            for word in tokens
        ]

        ids, segs = pack_pair(
            token_ids,
            max_len=max_len
        )

        all_ids.append(ids)
        all_segs.append(segs)
        all_labels.append(label)

    if not all_ids:
        return (
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0,), dtype=np.int32)
        )

    return (
        np.asarray(all_ids, dtype=np.int32),
        np.asarray(all_segs, dtype=np.int32),
        np.asarray(all_labels, dtype=np.int32)
    )


def topic_split():
    # Preserve the class order: animal, food, thing.
    categories = list(TOPICS.keys())

    train = {}
    heldout = {}

    for c, category in enumerate(categories):
        topic_words = TOPICS[category]

        # Even positions are used for training.
        train[c] = set(topic_words[::2])

        # Odd positions are reserved for testing.
        heldout[c] = set(topic_words[1::2])

    return {
        "train": train,
        "heldout": heldout
    }


def topic_set(stories, words, n, vocab, seed=0, max_len=32):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    categories = list(TOPICS.keys())

    # Map each topic word to its integer class index.
    topic_to_class = {}

    for c, category in enumerate(categories):
        for word in TOPICS[category]:
            topic_to_class[word] = c

    # Collect eligible sentences separately for each class.
    class_rows = {c: [] for c in range(3)}

    # Examine all tagged stories.
    for story in stories:
        for tokens, tags in story:

            # Keep sentences within the required length.
            if not (4 <= len(tokens) <= max_len - 2):
                continue

            # Find all topic words appearing in the sentence.
            found_words = [
                word for word in tokens
                if word in topic_to_class
            ]

            # At least one topic word is required.
            if not found_words:
                continue

            # All topic words must belong to exactly one class.
            present_classes = {
                topic_to_class[word]
                for word in found_words
            }

            if len(present_classes) != 1:
                continue

            c = next(iter(present_classes))

            # Every topic word must belong to the supplied
            # training or held-out vocabulary for this class.
            if not all(word in words[c] for word in found_words):
                continue

            class_rows[c].append((tokens, c))

    # Take up to n // 3 examples from each class.
    per_class = n // 3
    selected = []

    for c in range(3):
        rows = class_rows[c]

        order = rng.permutation(len(rows))

        for idx in order[:per_class]:
            selected.append(rows[idx])

    # Permute all selected rows.
    order = rng.permutation(len(selected))
    selected = [selected[i] for i in order]

    all_ids = []
    all_segs = []
    all_labels = []

    # Pack every selected sentence independently.
    for tokens, label in selected:

        token_ids = [
            vocab.get(word, UNK)
            for word in tokens
        ]

        ids, segs = pack_pair(
            token_ids,
            max_len=max_len
        )

        all_ids.append(ids)
        all_segs.append(segs)
        all_labels.append(label)

    if not all_ids:
        return (
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0,), dtype=np.int32)
        )

    return (
        np.asarray(all_ids, dtype=np.int32),
        np.asarray(all_segs, dtype=np.int32),
        np.asarray(all_labels, dtype=np.int32)
    )


def cls_accuracy(params, head, ids, segs, y):
    # Convert labels into a NumPy array.
    y = np.asarray(y, dtype=np.int32)

    if len(y) == 0:
        raise ValueError("Evaluation dataset must not be empty.")

    # Encode all sequences using the final encoder layer.
    h = encode_all(
        params,
        ids,
        segs,
        bs=256,
        layers=False
    )

    # Extract the [CLS] representation.
    cls_features = Tensor(h[:, 0, :])

    # Apply the classification head.
    logits = head(cls_features)

    # Get predicted class indices.
    predictions = logits.numpy().argmax(axis=-1)

    # Calculate classification accuracy.
    return float(np.mean(predictions == y))

# Step 25 - both_person_set
def both_person_set(stories, n, vocab, seed=0, max_len=32):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Collect eligible sentences into two groups.
    person = []
    none = []

    # ---------------------------------------------------------
    # 1. Identify person-only and neither-entity sentences.
    # ---------------------------------------------------------

    for story in stories:
        for tokens, tags in story:

            # Keep sentences with lengths between 4 and 14.
            if not (4 <= len(tokens) <= 14):
                continue

            has_person = 1 in tags
            has_place = 3 in tags

            # Person-only sentence: contains a person but no place.
            if has_person and not has_place:
                person.append(tokens)

            # Neither sentence: contains neither a person nor a place.
            elif not has_person and not has_place:
                none.append(tokens)

    # Validate the available sentence pools.
    if n > 0 and not person:
        raise ValueError("No eligible person-only sentences found.")

    if n > 1 and not none:
        raise ValueError("No eligible neither-entity sentences found.")

    # ---------------------------------------------------------
    # 2. Generate positive and negative sentence pairs.
    # ---------------------------------------------------------

    all_ids = []
    all_segs = []
    all_labels = []

    for i in range(n):

        if i % 2 == 0:
            # Positive example: both sentences name a person.
            A = person[int(rng.integers(len(person)))]
            B = person[int(rng.integers(len(person)))]

            label = 1

        else:
            # Negative example: randomly choose one of the
            # three permitted sentence-type combinations.
            choice = int(rng.integers(3))

            if choice == 0:
                # Person + Neither.
                A = person[int(rng.integers(len(person)))]
                B = none[int(rng.integers(len(none)))]

            elif choice == 1:
                # Neither + Person.
                A = none[int(rng.integers(len(none)))]
                B = person[int(rng.integers(len(person)))]

            else:
                # Neither + Neither.
                A = none[int(rng.integers(len(none)))]
                B = none[int(rng.integers(len(none)))]

            label = 0

        # Convert both sentences into vocabulary IDs.
        A_ids = [
            vocab.get(word, UNK)
            for word in A
        ]

        B_ids = [
            vocab.get(word, UNK)
            for word in B
        ]

        # Pack the sentence pair into BERT format.
        ids, segs = pack_pair(
            A_ids,
            B_ids,
            max_len=max_len
        )

        all_ids.append(ids)
        all_segs.append(segs)
        all_labels.append(label)

    # ---------------------------------------------------------
    # 3. Randomly permute the completed examples.
    # ---------------------------------------------------------

    if n == 0:
        return (
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0,), dtype=np.int32)
        )

    order = rng.permutation(n)

    return (
        np.asarray(all_ids, dtype=np.int32)[order],
        np.asarray(all_segs, dtype=np.int32)[order],
        np.asarray(all_labels, dtype=np.int32)[order]
    )


def same_story_set(enc, n, seed=0, max_len=32, min_gap=2):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    if n < 0:
        raise ValueError("n must be non-negative.")

    if min_gap < 1:
        raise ValueError("min_gap must be positive.")

    if n == 0:
        return (
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0,), dtype=np.int32)
        )

    if not enc:
        raise ValueError("enc must contain at least one story.")

    # At least one story must be long enough to generate positives.
    if not any(len(story) >= min_gap + 2 for story in enc):
        raise ValueError("No story has enough sentences for positive pairs.")

    # At least two non-empty stories are required for negative pairs.
    if n > 1 and sum(bool(story) for story in enc) < 2:
        raise ValueError("At least two non-empty stories are required.")

    all_ids = []
    all_segs = []
    all_labels = []

    # ---------------------------------------------------------
    # 1. Generate positive and negative sentence pairs.
    # ---------------------------------------------------------

    for i in range(n):

        while True:
            # Draw a random story.
            s = int(rng.integers(len(enc)))
            story = enc[s]

            # The story must contain enough sentences.
            if len(story) < min_gap + 2:
                continue

            # Draw sentence A.
            a = int(rng.integers(len(story) - min_gap))
            A = story[a][0]

            if i % 2 == 0:
                # Positive example:
                # B must come from the same story and must be
                # at least min_gap sentences after A.

                b = int(
                    rng.integers(a + min_gap, len(story))
                )

                B = story[b][0]
                label = 1

            else:
                # Negative example:
                # B must come from a different story.

                o = int(rng.integers(len(enc)))

                # Reject the same story or an empty story.
                if o == s or len(enc[o]) == 0:
                    continue

                other_story = enc[o]

                # Select a random sentence from the other story.
                b = int(rng.integers(len(other_story)))

                B = other_story[b][0]
                label = 0

            break

        # Pack the sentence pair.
        ids, segs = pack_pair(
            A,
            B,
            max_len=max_len
        )

        all_ids.append(ids)
        all_segs.append(segs)
        all_labels.append(label)

    # ---------------------------------------------------------
    # 2. Randomly permute all generated examples.
    # ---------------------------------------------------------

    order = rng.permutation(n)

    return (
        np.asarray(all_ids, dtype=np.int32)[order],
        np.asarray(all_segs, dtype=np.int32)[order],
        np.asarray(all_labels, dtype=np.int32)[order]
    )


def nsp_accuracy(params, ids, segs, y):
    # Convert labels into a NumPy array.
    y = np.asarray(y, dtype=np.int32)

    if len(y) == 0:
        raise ValueError("Evaluation dataset must not be empty.")

    # Encode the sentence pairs using the final encoder layer.
    h = encode_all(
        params,
        ids,
        segs,
        bs=256,
        layers=False
    )

    # Convert the final hidden representations into a Tensor.
    h_tensor = Tensor(h)

    # Apply the pretrained next-sentence prediction classifier.
    logits = nsp_logits(
        params,
        h_tensor
    ).numpy()

    # Obtain the predicted NSP labels.
    predictions = logits.argmax(axis=-1)

    # Return classification accuracy as a Python float.
    return float(np.mean(predictions == y))

# Step 26 - tag_set
def tag_set(enc, n, seed=0, max_len=32):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    if n < 0:
        raise ValueError("n must be non-negative.")

    # Collect sentences into entity-containing and entity-free groups.
    with_entities = []
    without_entities = []

    # ---------------------------------------------------------
    # 1. Separate eligible sentences into two groups.
    # ---------------------------------------------------------

    for story in enc:
        for ids, tags in story:

            # Keep sentences with lengths between 4 and max_len - 2.
            if not (4 <= len(ids) <= max_len - 2):
                continue

            # A sentence contains an entity if at least one tag is > 0.
            if np.any(tags > 0):
                with_entities.append((ids, tags))
            else:
                without_entities.append((ids, tags))

    # ---------------------------------------------------------
    # 2. Shuffle both groups independently.
    # ---------------------------------------------------------

    rng.shuffle(with_entities)
    rng.shuffle(without_entities)

    # ---------------------------------------------------------
    # 3. Select the required number of examples.
    # ---------------------------------------------------------

    # Aim for 75% entity-containing sentences.
    # Do not select more than are available.
    m = min(int(n * 0.75), len(with_entities))

    # Select m entity-containing sentences.
    selected_entities = with_entities[:m]

    # Fill the remaining positions with entity-free sentences.
    remaining = n - m
    selected_without = without_entities[:remaining]

    # Combine the two groups.
    selected = selected_entities + selected_without

    # ---------------------------------------------------------
    # 4. Shuffle the combined selection.
    # ---------------------------------------------------------

    order = rng.permutation(len(selected))
    selected = [selected[i] for i in order]

    # ---------------------------------------------------------
    # 5. Pack sentences and construct their tag arrays.
    # ---------------------------------------------------------

    all_ids = []
    all_segs = []
    all_tags = []

    for sentence_ids, sentence_tags in selected:

        # Pack each sentence independently.
        ids, segs = pack_pair(
            sentence_ids,
            max_len=max_len
        )

        # Initialize all positions with -1.
        # -1 means that the position must be ignored by the loss.
        packed_tags = np.full(
            max_len,
            -1,
            dtype=np.int32
        )

        # Position 0 is [CLS], so actual sentence tags begin at 1.
        # The final [SEP] and padding positions remain -1.
        length = len(sentence_ids)

        packed_tags[1:length + 1] = sentence_tags

        all_ids.append(ids)
        all_segs.append(segs)
        all_tags.append(packed_tags)

    # ---------------------------------------------------------
    # 6. Return fixed-length NumPy arrays.
    # ---------------------------------------------------------

    if not all_ids:
        return (
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0, max_len), dtype=np.int32),
            np.empty((0, max_len), dtype=np.int32)
        )

    return (
        np.asarray(all_ids, dtype=np.int32),
        np.asarray(all_segs, dtype=np.int32),
        np.asarray(all_tags, dtype=np.int32)
    )


def tag_loss(head, h, tags):
    # Apply the tagging head to every token representation.
    # h: (B, T, d)
    # logits: (B, T, 5)
    logits = head(h)

    # Identify positions containing valid BIO tags.
    # Special tokens and padding have tag -1.
    valid = tags >= 0

    # Replace ignored labels with zero temporarily so that
    # one-hot conversion receives valid class indices.
    safe_tags = tags.maximum(0)

    # Calculate log probabilities over the five BIO classes.
    log_probs = logits.log_softmax(axis=-1)

    # Convert target tags into one-hot representations.
    targets = safe_tags.one_hot(5).cast(logits.dtype)

    # Calculate negative log-likelihood for every position.
    losses = -(log_probs * targets).sum(axis=-1)

    # Convert the validity mask into floating-point weights.
    weights = valid.cast(logits.dtype)

    # Calculate the mean loss over valid token positions only.
    numerator = (losses * weights).sum()
    denominator = weights.sum().maximum(1.0)

    return numerator / denominator

# Step 27 - viterbi_bio
def bio_to_spans(tags):
    # Store the extracted entity spans.
    spans = []

    # Track the currently active entity.
    current_type = None
    start = None

    # Helper to close an active entity.
    def close_span(end):
        nonlocal current_type, start

        if current_type is not None:
            spans.append((current_type, start, end))

        current_type = None
        start = None

    # Process tags from left to right.
    for i, tag_id in enumerate(tags):

        tag_id = int(tag_id)

        # O: no entity at this position.
        if tag_id == 0:
            close_span(i - 1)

        # B-Per: always starts a new person entity.
        elif tag_id == 1:
            close_span(i - 1)
            current_type = "Per"
            start = i

        # I-Per: continue the current person entity,
        # or start a new one if no matching entity exists.
        elif tag_id == 2:
            if current_type != "Per":
                close_span(i - 1)
                current_type = "Per"
                start = i

        # B-Loc: always starts a new location entity.
        elif tag_id == 3:
            close_span(i - 1)
            current_type = "Loc"
            start = i

        # I-Loc: continue the current location entity,
        # or start a new one if no matching entity exists.
        elif tag_id == 4:
            if current_type != "Loc":
                close_span(i - 1)
                current_type = "Loc"
                start = i

        else:
            raise ValueError(f"Invalid BIO tag index: {tag_id}")

    # Close any entity remaining at the end of the sequence.
    close_span(len(tags) - 1)

    return spans


def viterbi_bio(log_probs):
    # Convert the input into a NumPy float array.
    log_probs = np.asarray(log_probs, dtype=np.float64)

    # Input shape: (T, 5).
    if log_probs.ndim != 2 or log_probs.shape[1] != 5:
        raise ValueError("log_probs must have shape (T, 5).")

    T = len(log_probs)

    # Handle an empty sequence.
    if T == 0:
        return []

    # ---------------------------------------------------------
    # 1. Define the legal BIO transitions.
    # ---------------------------------------------------------

    # A transition is legal when the current tag can follow
    # the previous tag.
    #
    # Tag 2 (I-Per) can only follow tag 1 or 2.
    # Tag 4 (I-Loc) can only follow tag 3 or 4.
    # All other transitions are permitted.

    allowed = np.ones((5, 5), dtype=bool)

    allowed[:, 2] = False
    allowed[1, 2] = True
    allowed[2, 2] = True

    allowed[:, 4] = False
    allowed[3, 4] = True
    allowed[4, 4] = True

    # ---------------------------------------------------------
    # 2. Initialize the dynamic programming table.
    # ---------------------------------------------------------

    dp = np.full((T, 5), -np.inf, dtype=np.float64)

    # Store the best previous tag for every position and tag.
    back = np.zeros((T, 5), dtype=np.int32)

    # The first tag cannot be I-Per (2) or I-Loc (4).
    dp[0, [0, 1, 3]] = log_probs[0, [0, 1, 3]]

    # ---------------------------------------------------------
    # 3. Perform the Viterbi forward pass.
    # ---------------------------------------------------------

    for t in range(1, T):

        for current_tag in range(5):

            # Consider only legal previous tags.
            previous_tags = np.flatnonzero(
                allowed[:, current_tag]
            )

            # Calculate the candidate scores.
            scores = (
                dp[t - 1, previous_tags]
                + log_probs[t, current_tag]
            )

            # Select the highest-scoring previous tag.
            best_idx = int(np.argmax(scores))
            best_previous = int(previous_tags[best_idx])

            # Save the best score and backpointer.
            dp[t, current_tag] = scores[best_idx]
            back[t, current_tag] = best_previous

    # ---------------------------------------------------------
    # 4. Backtrack to recover the best tag sequence.
    # ---------------------------------------------------------

    # Select the highest-scoring final tag.
    last_tag = int(np.argmax(dp[T - 1]))

    # Reconstruct the complete sequence.
    path = [0] * T
    path[T - 1] = last_tag

    for t in range(T - 1, 0, -1):
        path[t - 1] = int(back[t, path[t]])

    return path


def span_f1(true_seqs, pred_seqs):
    # Ensure both collections contain the same number of sequences.
    if len(true_seqs) != len(pred_seqs):
        raise ValueError(
            "true_seqs and pred_seqs must have equal lengths."
        )

    true_count = 0
    pred_count = 0
    true_positive = 0

    # Evaluate entity spans independently within each sequence.
    for true_tags, pred_tags in zip(true_seqs, pred_seqs):

        # Convert BIO tags into entity spans.
        true_spans = set(bio_to_spans(true_tags))
        pred_spans = set(bio_to_spans(pred_tags))

        # Count all true and predicted entities.
        true_count += len(true_spans)
        pred_count += len(pred_spans)

        # An entity is correct only when its type, start
        # and end positions all match.
        true_positive += len(true_spans & pred_spans)

    # Calculate precision.
    precision = (
        true_positive / pred_count
        if pred_count > 0
        else 0.0
    )

    # Calculate recall.
    recall = (
        true_positive / true_count
        if true_count > 0
        else 0.0
    )

    # Calculate F1, handling the zero-denominator case.
    if precision + recall > 0:
        f1 = (
            2.0 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return (
        float(f1),
        float(precision),
        float(recall)
    )


def tag_f1(params, head, ids, segs, tags):
    # Convert the target tags into a NumPy array.
    tags = np.asarray(tags, dtype=np.int32)

    # ---------------------------------------------------------
    # 1. Encode the input sequences.
    # ---------------------------------------------------------

    h = encode_all(
        params,
        ids,
        segs,
        bs=256,
        layers=False
    )

    # Convert the final encoder representations into a Tensor.
    h_tensor = Tensor(h, dtype=dtypes.float32)

    # ---------------------------------------------------------
    # 2. Calculate tagging logits and log probabilities.
    # ---------------------------------------------------------

    logits = head(h_tensor)

    log_probs = logits.log_softmax(axis=-1).numpy()

    raw_predictions = logits.numpy().argmax(axis=-1)

    # ---------------------------------------------------------
    # 3. Remove special tokens and padding.
    # ---------------------------------------------------------

    valid = tags >= 0

    true_seqs = []
    viterbi_predictions = []
    raw_predictions_valid = []

    for i in range(len(tags)):

        # Keep only positions containing actual sentence tags.
        mask = valid[i]

        true_seq = tags[i, mask]
        log_prob_seq = log_probs[i, mask]
        raw_seq = raw_predictions[i, mask]

        # Decode the best legal BIO sequence.
        decoded_seq = viterbi_bio(log_prob_seq)

        true_seqs.append(true_seq.tolist())
        viterbi_predictions.append(decoded_seq)
        raw_predictions_valid.append(raw_seq.tolist())

    # ---------------------------------------------------------
    # 4. Calculate entity-level F1 for both predictions.
    # ---------------------------------------------------------

    viterbi_f1, _, _ = span_f1(
        true_seqs,
        viterbi_predictions
    )

    raw_f1, _, _ = span_f1(
        true_seqs,
        raw_predictions_valid
    )

    return float(viterbi_f1), float(raw_f1)


def tag_text(params, head, vocab, text):
    # Tokenize the input text after converting it to lowercase.
    tokens = TOKEN.findall(text.lower())

    # An empty input produces an empty output.
    if not tokens:
        return []

    # Limit the sentence to the number of tokens that fit.
    max_tokens = params["max_len"] - 2
    tokens = tokens[:max_tokens]

    # Convert words to vocabulary IDs.
    # Unknown words receive the UNK token.
    token_ids = [
        vocab.get(word, UNK)
        for word in tokens
    ]

    # Pack the sentence into BERT input format.
    ids, segs = pack_pair(
        token_ids,
        max_len=params["max_len"]
    )

    # ---------------------------------------------------------
    # 1. Encode the sentence.
    # ---------------------------------------------------------

    h = encode_all(
        params,
        ids[None, :],
        segs[None, :],
        bs=256,
        layers=False
    )

    # ---------------------------------------------------------
    # 2. Apply the tagging head.
    # ---------------------------------------------------------

    h_tensor = Tensor(h, dtype=dtypes.float32)

    logits = head(h_tensor)

    log_probs = logits.log_softmax(axis=-1).numpy()[0]

    # ---------------------------------------------------------
    # 3. Decode only the actual sentence-token positions.
    # ---------------------------------------------------------

    # Position 0 is [CLS], so sentence tokens begin at position 1.
    token_log_probs = log_probs[1:len(tokens) + 1]

    predicted_tags = viterbi_bio(token_log_probs)

    # ---------------------------------------------------------
    # 4. Return tokens with their predicted BIO tag names.
    # ---------------------------------------------------------

    return [
        (token, TAGS[tag_id])
        for token, tag_id in zip(tokens, predicted_tags)
    ]

# Step 28 - make_qa
def qa_pools(stories, max_tok=12):
    """
    Build two datasets:
      pool: sentences with exactly one person and one place.
      filler: sentences containing no entities.
    """

    pool = []
    filler = []

    for story in stories:
        for tokens, tags in story:

            spans = bio_to_spans(tags)

            if len(tokens) > max_tok:
                continue

            person_spans = [
                (entity_type, start, end)
                for entity_type, start, end in spans
                if entity_type == "Per"
            ]

            place_spans = [
                (entity_type, start, end)
                for entity_type, start, end in spans
                if entity_type == "Loc"
            ]

            # Exactly one person and one location.
            if len(person_spans) == 1 and len(place_spans) == 1:

                pool.append((
                    list(tokens),
                    person_spans[0],
                    place_spans[0]
                ))

            # At least three tokens and no entities.
            elif len(tokens) >= 3 and not spans:

                filler.append(list(tokens))

    return pool, filler


def make_qa(pool, filler, vocab, n, seed=0, max_len=32):
    """
    Generate question-passage pairs with known answer spans.

    Returns:
        ids:   (n, max_len) int32
        segs:  (n, max_len) int32
        start: (n,) int32
        end:   (n,) int32
    """

    rng = np.random.default_rng(seed)

    if n < 0:
        raise ValueError("n must be non-negative.")

    if max_len < 19:
        raise ValueError("max_len must be at least 19.")

    if n > 0 and not pool:
        raise ValueError("pool must contain at least one sentence.")

    if n > 0 and not filler:
        raise ValueError("filler must contain at least one sentence.")

    all_ids = []
    all_segs = []
    all_start = []
    all_end = []

    for _ in range(n):

        # -------------------------------------------------
        # 1. Select a pool sentence and a filler sentence.
        # -------------------------------------------------

        pool_idx = int(rng.integers(len(pool)))

        pool_tokens, person_span, place_span = pool[pool_idx]

        filler_idx = int(rng.integers(len(filler)))

        filler_tokens = list(filler[filler_idx])

        # -------------------------------------------------
        # 2. Extract the answer spans.
        # -------------------------------------------------

        # Spans are stored as:
        # (entity_type, start, end)

        person_type, person_start, person_end = person_span
        place_type, place_start, place_end = place_span

        if person_type != "Per" or place_type != "Loc":
            raise ValueError("Invalid person or location span.")

        # Choose a question and its corresponding answer.
        if rng.random() < 0.5:

            question = ["where", "is", "this", "?"]

            answer_start = place_start
            answer_end = place_end

        else:

            question = ["who", "is", "here", "?"]

            answer_start = person_start
            answer_end = person_end

        # -------------------------------------------------
        # 3. Convert the question to vocabulary IDs.
        # -------------------------------------------------

        question_ids = [
            vocab.get(word, UNK)
            for word in question
        ]

        # -------------------------------------------------
        # 4. Build the passage in random sentence order.
        # -------------------------------------------------

        # Reserve enough space for:
        # [CLS] question [SEP] passage [SEP]

        passage_capacity = max_len - len(question_ids) - 3

        # Preserve the entire pool sentence because it
        # contains the answer.
        if len(pool_tokens) > passage_capacity:
            raise ValueError(
                "The pool sentence does not fit in max_len."
            )

        # Truncate only the filler if necessary.
        filler_tokens = filler_tokens[
            :passage_capacity - len(pool_tokens)
        ]

        if rng.random() < 0.5:

            # Pool sentence followed by filler.
            passage = list(pool_tokens) + filler_tokens

            pool_offset = 0

        else:

            # Filler sentence followed by pool sentence.
            passage = filler_tokens + list(pool_tokens)

            pool_offset = len(filler_tokens)

        # -------------------------------------------------
        # 5. Convert passage words to vocabulary IDs.
        # -------------------------------------------------

        passage_ids = [
            vocab.get(word, UNK)
            for word in passage
        ]

        # -------------------------------------------------
        # 6. Pack the question and passage.
        # -------------------------------------------------

        ids, segs = pack_pair(
            question_ids,
            passage_ids,
            max_len=max_len
        )

        # -------------------------------------------------
        # 7. Calculate the answer's packed positions.
        # -------------------------------------------------

        # Packed format:
        #
        # [CLS] question [SEP] passage [SEP] [PAD] ...
        #
        # Passage begins after CLS, question and SEP.

        passage_start = len(question_ids) + 2

        # Locate the pool sentence inside the passage.
        answer_offset = passage_start + pool_offset

        # BIO span indices are inclusive and sentence-relative.
        start = answer_offset + answer_start
        end = answer_offset + answer_end

        # Verify that the answer remains inside the passage.
        if (
            start < passage_start
            or end >= passage_start + len(passage)
            or end >= max_len
            or start > end
        ):
            raise ValueError("Invalid or truncated answer span.")

        all_ids.append(ids)
        all_segs.append(segs)
        all_start.append(start)
        all_end.append(end)

    # -------------------------------------------------
    # 8. Return the complete dataset.
    # -------------------------------------------------

    return (
        np.asarray(all_ids, dtype=np.int32).reshape(n, max_len),
        np.asarray(all_segs, dtype=np.int32).reshape(n, max_len),
        np.asarray(all_start, dtype=np.int32),
        np.asarray(all_end, dtype=np.int32)
    )


def passage_mask(ids, segs):
    """
    True only for passage tokens.
    Excludes SEP and PAD.
    """

    ids = np.asarray(ids)
    segs = np.asarray(segs)

    return (
        (segs == 1)
        & (ids != SEP)
        & (ids != PAD)
    )


def qa_loss(head_s, head_e, h, start, end, mask):
    """
    Question-answering loss:
    start-position cross-entropy + end-position cross-entropy.
    """

    B, T, d = h.shape

    # -------------------------------------------------
    # 1. Calculate start and end logits.
    # -------------------------------------------------

    start_logits = head_s(h).reshape(B, T)
    end_logits = head_e(h).reshape(B, T)

    # -------------------------------------------------
    # 2. Mask positions outside the passage.
    # -------------------------------------------------

    mask_float = mask.cast(start_logits.dtype)

    start_logits = (
        start_logits + (1.0 - mask_float) * (-1e9)
    )

    end_logits = (
        end_logits + (1.0 - mask_float) * (-1e9)
    )

    # -------------------------------------------------
    # 3. Start-position cross-entropy.
    # -------------------------------------------------

    start_log_probs = start_logits.log_softmax(axis=-1)

    start_targets = start.one_hot(T).cast(
        start_log_probs.dtype
    )

    start_loss = -(
        start_log_probs * start_targets
    ).sum(axis=-1).mean()

    # -------------------------------------------------
    # 4. End-position cross-entropy.
    # -------------------------------------------------

    end_log_probs = end_logits.log_softmax(axis=-1)

    end_targets = end.one_hot(T).cast(
        end_log_probs.dtype
    )

    end_loss = -(
        end_log_probs * end_targets
    ).sum(axis=-1).mean()

    # -------------------------------------------------
    # 5. Combined QA loss.
    # -------------------------------------------------

    return start_loss + end_loss

# Step 29 - best_span
def best_span(s, e, mask, max_span=4):
    # Convert inputs into NumPy arrays.
    s = np.asarray(s)
    e = np.asarray(e)
    mask = np.asarray(mask, dtype=bool)

    # Validate the input dimensions.
    if s.ndim != 1 or e.ndim != 1 or mask.ndim != 1:
        raise ValueError("s, e and mask must be 1D arrays.")

    if len(s) != len(e) or len(s) != len(mask):
        raise ValueError("s, e and mask must have equal lengths.")

    if max_span <= 0:
        return (0, 0)

    T = len(s)

    # Initialize the best score and answer positions.
    best_score = -np.inf
    best_i, best_j = 0, 0

    # ---------------------------------------------------------
    # Search for the highest-scoring valid answer span.
    # ---------------------------------------------------------

    # Iterate over possible start positions in ascending order.
    for i in range(T):

        # The start position must belong to the passage.
        if not mask[i]:
            continue

        # Limit the end position to the maximum answer length.
        end_limit = min(T, i + max_span)

        # Iterate over valid end positions in ascending order.
        for j in range(i, end_limit):

            # The end position must also belong to the passage.
            if not mask[j]:
                continue

            # Calculate the combined start and end score.
            score = float(s[i]) + float(e[j])

            # Update only when the score is strictly greater.
            # This preserves the earliest i, then earliest j,
            # when multiple spans have equal scores.
            if score > best_score:
                best_score = score
                best_i = i
                best_j = j

    # Return (0, 0) if no valid answer span exists.
    if best_score == -np.inf:
        return (0, 0)

    return (int(best_i), int(best_j))


def qa_accuracy(params, head_s, head_e, ids, segs, start, end):
    # Convert the input arrays into NumPy arrays.
    ids = np.asarray(ids, dtype=np.int32)
    segs = np.asarray(segs, dtype=np.int32)
    start = np.asarray(start, dtype=np.int32)
    end = np.asarray(end, dtype=np.int32)

    # Validate the batch dimensions.
    if ids.ndim != 2 or segs.ndim != 2:
        raise ValueError("ids and segs must be 2D arrays.")

    if ids.shape != segs.shape:
        raise ValueError("ids and segs must have identical shapes.")

    if start.shape != (len(ids),) or end.shape != (len(ids),):
        raise ValueError("start and end must have shape (B,).")

    if len(ids) == 0:
        return 0.0

    # ---------------------------------------------------------
    # 1. Encode the question-passage pairs.
    # ---------------------------------------------------------

    h = encode_all(
        params,
        ids,
        segs,
        bs=256,
        layers=False
    )

    # Convert the final encoder representations into a Tensor.
    h_tensor = Tensor(h)

    # Get the batch size and sequence length.
    B, T, d = h.shape

    # ---------------------------------------------------------
    # 2. Calculate start and end position scores.
    # ---------------------------------------------------------

    # Each head produces one score for every sequence position.
    start_logits = head_s(h_tensor).reshape(B, T).numpy()

    end_logits = head_e(h_tensor).reshape(B, T).numpy()

    # ---------------------------------------------------------
    # 3. Identify valid passage positions.
    # ---------------------------------------------------------

    mask = passage_mask(ids, segs)

    # ---------------------------------------------------------
    # 4. Decode the best answer span for each example.
    # ---------------------------------------------------------

    correct = 0

    for i in range(B):

        predicted_start, predicted_end = best_span(
            start_logits[i],
            end_logits[i],
            mask[i]
        )

        # Exact-span evaluation:
        # Both the predicted start and end must match.
        if (
            predicted_start == int(start[i])
            and predicted_end == int(end[i])
        ):
            correct += 1

    # ---------------------------------------------------------
    # 5. Return exact-span accuracy.
    # ---------------------------------------------------------

    return float(correct / B)

# Step 30 - tiny_bert_report
# ---------------------------------------------------------
# DEFAULT TRAINING-LABEL BUDGETS
# ---------------------------------------------------------

import time

DEFAULT_BUDGETS = {
    "mention": (128,),
    "topic": (128,),
    "both_person": (256,),
    "same_story": (),
    "tag": (128,),
    "qa": (128,)
}


# ---------------------------------------------------------
# DEFAULT TRAINING CONDITIONS
# ---------------------------------------------------------

DEFAULT_CONDITIONS = {
    "mention": ("scratch", "fine"),
    "topic": ("scratch", "fine"),
    "both_person": ("scratch", "fine"),
    "same_story": ("scratch", "fine"),
    "tag": ("scratch", "frozen", "fine"),
    "qa": ("scratch", "frozen", "fine")
}


# ---------------------------------------------------------
# COMPARE TRAINING CONDITIONS
# ---------------------------------------------------------

def compare_conditions(
    tuner, weights, pre_state, rand_state,
    head_params, head_state, arrays, budgets, evaluate,
    conditions=("scratch", "frozen", "fine"),
    steps=150, batch=32, lr_enc=5e-4,
    lr_head=1e-3, seed=0
):

    results = {}

    arrays = tuple(np.asarray(a) for a in arrays)

    if not arrays:
        raise ValueError("arrays must contain training data.")

    n_available = len(arrays[0])

    if any(len(a) != n_available for a in arrays):
        raise ValueError("All training arrays must have equal lengths.")

    if batch <= 0:
        raise ValueError("batch must be positive.")

    for n in budgets:

        n = int(n)

        if n <= 0:
            continue

        limit = min(n, n_available)

        if limit == 0:
            continue

        results[n] = {}

        def batch_fn(rng):
            idx = rng.integers(
                0,
                limit,
                size=batch
            )
            return tuple(a[idx] for a in arrays)

        for condition in conditions:

            if condition == "scratch":
                restore(weights, rand_state)
                encoder_lr = lr_head

            elif condition == "frozen":
                restore(weights, pre_state)
                encoder_lr = 0.0

            elif condition == "fine":
                restore(weights, pre_state)
                encoder_lr = lr_enc

            else:
                raise ValueError(
                    f"Unknown training condition: {condition}"
                )

            restore(head_params, head_state)

            tuner.run(
                batch_fn,
                steps,
                encoder_lr,
                lr_head,
                seed=seed
            )

            results[n][condition] = evaluate()

    return results


# ---------------------------------------------------------
# NEAREST WORDS
# ---------------------------------------------------------

def nearest_words(params, vocab, word, k=6):

    if k <= 0 or word not in vocab:
        return []

    E = params["tok"].numpy()

    word_idx = vocab[word]

    if word_idx < 5 or word_idx >= len(E):
        return []

    norms = np.linalg.norm(E, axis=1)

    denominator = norms * norms[word_idx] + 1e-12

    similarities = (
        E @ E[word_idx]
    ) / denominator

    # Exclude special tokens and the word itself.
    similarities[:5] = -np.inf
    similarities[word_idx] = -np.inf

    order = np.argsort(-similarities, kind="stable")

    id_to_word = {
        idx: token
        for token, idx in vocab.items()
    }

    results = []

    for idx in order:

        if not np.isfinite(similarities[idx]):
            continue

        if idx in id_to_word:
            results.append(id_to_word[idx])

        if len(results) >= k:
            break

    return results


# ---------------------------------------------------------
# MAIN TINY BERT REPORT
# ---------------------------------------------------------

def tiny_bert_report(
    seed=0,
    vocab_size=1200,
    n_names=100,
    d_model=64,
    n_layers=3,
    n_heads=4,
    max_len=32,
    stories=(6000, 2000, 2000),
    pretrain_steps=2000,
    finetune_steps=150,
    probe_steps=300,
    budgets=None,
    conditions=None
):

    # =====================================================
    # 0. VALIDATE CONFIGURATION
    # =====================================================

    if len(stories) != 3:
        raise ValueError(
            "stories must contain pretraining, training and test sizes."
        )

    n_pre, n_train, n_test = map(int, stories)

    if min(n_pre, n_train, n_test) <= 0:
        raise ValueError("All story-slice sizes must be positive.")

    if pretrain_steps <= 0 or finetune_steps <= 0:
        raise ValueError("Training steps must be positive.")

    if max_len < 19:
        raise ValueError(
            "max_len must be at least 19 for the QA dataset."
        )

    seconds = {}

    # =====================================================
    # 1. PREPARE CORPUS AND ENTITY SPLITS
    # =====================================================

    block_start = time.perf_counter()

    entities = entity_lists(
        k=n_names,
        seed=seed + 101
    )

    all_names, all_places = entities["all"]
    train_names, train_places = entities["train"]
    heldout_names, heldout_places = entities["heldout"]

    pre_raw = load_stories(0, n_pre)

    train_raw = load_stories(
        n_pre,
        n_pre + n_train
    )

    test_raw = load_stories(
        n_pre + n_train,
        n_pre + n_train + n_test
    )

    pre_stories = plant_entities(
        pre_raw,
        all_names,
        all_places,
        seed=seed + 211
    )

    train_stories = plant_entities(
        train_raw,
        train_names,
        train_places,
        seed=seed + 223
    )

    test_stories = plant_entities(
        test_raw,
        heldout_names,
        heldout_places,
        seed=seed + 227
    )

    vocab = build_vocab(
        pre_stories,
        max_size=vocab_size
    )

    actual_vocab_size = len(vocab)

    pre_enc = encode_stories(pre_stories, vocab)
    train_enc = encode_stories(train_stories, vocab)
    test_enc = encode_stories(test_stories, vocab)

    seconds["data"] = time.perf_counter() - block_start

    # =====================================================
    # 2. INITIALIZE AND PRETRAIN ENCODER
    # =====================================================

    block_start = time.perf_counter()

    params = init_params(
        actual_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=max_len,
        seed=seed + 307
    )

    history = pretrain(
        params,
        pre_enc,
        steps=pretrain_steps,
        batch=32,
        lr=2e-3,
        k=8,
        rate=0.25,
        warmup=100,
        seed=seed + 401,
        log_every=250
    )

    seconds["pretrain"] = time.perf_counter() - block_start

    # =====================================================
    # 3. MEASURE PRETRAINING
    # =====================================================

    block_start = time.perf_counter()

    metrics = mlm_metrics(
        params,
        pre_enc,
        n=512,
        seed=seed + 503,
        k=5,
        rate=0.15
    )

    future = future_context_gain(
        params,
        pre_enc,
        n=400,
        seed=seed + 509
    )

    book_cloze = cloze_topk(
        params,
        vocab,
        "Let's make [MASK] chicken! [SEP] It [MASK] great with orange sauce.",
        topk=5
    )

    story_cloze = cloze_topk(
        params,
        vocab,
        "the little girl went to the [MASK] to play with her [MASK] .",
        topk=5
    )

    cloze = {
        "book": book_cloze,
        "story": story_cloze
    }

    neighbours = {}

    for word in ("sad", "happy", "dog", "cake", "ran"):

        if word in vocab:
            neighbours[word] = nearest_words(
                params,
                vocab,
                word,
                k=6
            )

    seconds["measurements"] = time.perf_counter() - block_start

    # =====================================================
    # 4. ELMO-STYLE LAYER PROBES
    # =====================================================

    block_start = time.perf_counter()

    probes = layer_probes(
        params,
        vocab,
        train_stories,
        test_stories,
        seed=seed + 601,
        steps=probe_steps
    )

    seconds["probes"] = time.perf_counter() - block_start

    # =====================================================
    # 5. SAVE ENCODER CHECKPOINTS
    # =====================================================

    weights = encoder_parameters(params)

    pre_state = snapshot(weights)

    rand_params = init_params(
        actual_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=max_len,
        seed=seed + 701
    )

    rand_state = snapshot(
        encoder_parameters(rand_params)
    )

    # =====================================================
    # 6. PRETRAINED NSP ZERO-SHOT EVALUATION
    # =====================================================

    zero_ids, zero_segs, zero_y = same_story_set(
        test_enc,
        n=512,
        seed=seed + 809,
        max_len=max_len,
        min_gap=2
    )

    nsp_zero_shot = nsp_accuracy(
        params,
        zero_ids,
        zero_segs,
        zero_y
    )

    # =====================================================
    # 7. CONFIGURE BUDGETS AND CONDITIONS
    # =====================================================

    effective_budgets = dict(DEFAULT_BUDGETS)

    if budgets is not None:
        effective_budgets.update(budgets)

    effective_conditions = dict(DEFAULT_CONDITIONS)

    if conditions is not None:
        effective_conditions.update(conditions)

    # IMPORTANT FIX:
    # Only add tasks that actually have a non-empty budget.
    # Empty tasks such as topic or same_story must not appear
    # in the returned heads dictionary.

    heads = {}

    def maximum_budget(task):

        values = effective_budgets.get(task, ())

        return max(values) if values else 0

    def evaluation_size(task):

        return max(512, maximum_budget(task))

    d = d_model

    # =====================================================
    # HELPER: RUN ONE HEAD
    # =====================================================

    def run_head(
        task,
        head_params,
        head_state,
        loss_fn,
        arrays,
        evaluate,
        head_seed
    ):

        tuner = FineTuner(
            params,
            head_params,
            loss_fn
        )

        return compare_conditions(
            tuner=tuner,
            weights=weights,
            pre_state=pre_state,
            rand_state=rand_state,
            head_params=head_params,
            head_state=head_state,
            arrays=arrays,
            budgets=effective_budgets.get(task, ()),
            evaluate=evaluate,
            conditions=effective_conditions.get(
                task,
                DEFAULT_CONDITIONS[task]
            ),
            steps=finetune_steps,
            batch=32,
            lr_enc=5e-4,
            lr_head=1e-3,
            seed=seed + head_seed
        )

    # =====================================================
    # 8. TASK A — SENTENCE ENTITY-MENTION CLASSIFICATION
    # =====================================================

    block_start = time.perf_counter()

    mention_n = maximum_budget("mention")

    if mention_n > 0:

        mention_train = mention_set(
            train_stories,
            mention_n,
            vocab,
            seed=seed + 1001,
            max_len=max_len
        )

        mention_test = mention_set(
            test_stories,
            evaluation_size("mention"),
            vocab,
            seed=seed + 1002,
            max_len=max_len
        )

        tr_ids, tr_segs, tr_y = mention_train
        te_ids, te_segs, te_y = mention_test

        if len(tr_ids) == 0 or len(te_ids) == 0:
            raise ValueError("Mention classification data is empty.")

        mention_head = MLPHead(
            d,
            4,
            seed=seed + 1101
        )

        hp = mention_head.parameters()
        hs = snapshot(hp)

        mention_loss = lambda layers, y: (
            mention_head(layers[-1][:, 0, :])
            .sparse_categorical_crossentropy(y)
        )

        heads["mention"] = run_head(
            "mention",
            hp,
            hs,
            mention_loss,
            (tr_ids, tr_segs, tr_y),
            lambda: cls_accuracy(
                params,
                mention_head,
                te_ids,
                te_segs,
                te_y
            ),
            head_seed=1201
        )

    seconds["mention"] = time.perf_counter() - block_start

    # =====================================================
    # 9. TASK A — TOPIC CLASSIFICATION
    # =====================================================

    block_start = time.perf_counter()

    topic_n = maximum_budget("topic")

    if topic_n > 0:

        split = topic_split()

        topic_train = topic_set(
            train_stories,
            split["train"],
            topic_n,
            vocab,
            seed=seed + 1301,
            max_len=max_len
        )

        topic_test = topic_set(
            test_stories,
            split["heldout"],
            evaluation_size("topic"),
            vocab,
            seed=seed + 1302,
            max_len=max_len
        )

        tr_ids, tr_segs, tr_y = topic_train
        te_ids, te_segs, te_y = topic_test

        if len(tr_ids) == 0 or len(te_ids) == 0:
            raise ValueError("Topic classification data is empty.")

        topic_head = MLPHead(
            d,
            3,
            seed=seed + 1401
        )

        hp = topic_head.parameters()
        hs = snapshot(hp)

        topic_loss = lambda layers, y: (
            topic_head(layers[-1][:, 0, :])
            .sparse_categorical_crossentropy(y)
        )

        heads["topic"] = run_head(
            "topic",
            hp,
            hs,
            topic_loss,
            (tr_ids, tr_segs, tr_y),
            lambda: cls_accuracy(
                params,
                topic_head,
                te_ids,
                te_segs,
                te_y
            ),
            head_seed=1501
        )

    seconds["topic"] = time.perf_counter() - block_start

    # =====================================================
    # 10. TASK B — BOTH SENTENCES NAME A PERSON
    # =====================================================

    block_start = time.perf_counter()

    both_n = maximum_budget("both_person")

    if both_n > 0:

        both_train = both_person_set(
            train_stories,
            both_n,
            vocab,
            seed=seed + 1601,
            max_len=max_len
        )

        both_test = both_person_set(
            test_stories,
            evaluation_size("both_person"),
            vocab,
            seed=seed + 1602,
            max_len=max_len
        )

        tr_ids, tr_segs, tr_y = both_train
        te_ids, te_segs, te_y = both_test

        both_head = MLPHead(
            d,
            2,
            seed=seed + 1701
        )

        hp = both_head.parameters()
        hs = snapshot(hp)

        both_loss = lambda layers, y: (
            both_head(layers[-1][:, 0, :])
            .sparse_categorical_crossentropy(y)
        )

        heads["both_person"] = run_head(
            "both_person",
            hp,
            hs,
            both_loss,
            (tr_ids, tr_segs, tr_y),
            lambda: cls_accuracy(
                params,
                both_head,
                te_ids,
                te_segs,
                te_y
            ),
            head_seed=1801
        )

    seconds["both_person"] = time.perf_counter() - block_start

    # =====================================================
    # 11. TASK B — SAME-STORY CLASSIFICATION
    # =====================================================

    block_start = time.perf_counter()

    same_n = maximum_budget("same_story")

    if same_n > 0:

        same_train = same_story_set(
            train_enc,
            same_n,
            seed=seed + 1901,
            max_len=max_len,
            min_gap=2
        )

        tr_ids, tr_segs, tr_y = same_train

        same_head = MLPHead(
            d,
            2,
            seed=seed + 2001
        )

        hp = same_head.parameters()
        hs = snapshot(hp)

        same_loss = lambda layers, y: (
            same_head(layers[-1][:, 0, :])
            .sparse_categorical_crossentropy(y)
        )

        heads["same_story"] = run_head(
            "same_story",
            hp,
            hs,
            same_loss,
            (tr_ids, tr_segs, tr_y),
            lambda: cls_accuracy(
                params,
                same_head,
                zero_ids,
                zero_segs,
                zero_y
            ),
            head_seed=2101
        )

    seconds["same_story"] = time.perf_counter() - block_start

    # =====================================================
    # 12. TASK C — BIO ENTITY TAGGING
    # =====================================================

    block_start = time.perf_counter()

    tag_n = maximum_budget("tag")

    book_sentence = []

    if tag_n > 0:

        tag_train = tag_set(
            train_enc,
            tag_n,
            seed=seed + 2201,
            max_len=max_len
        )

        tag_test = tag_set(
            test_enc,
            evaluation_size("tag"),
            seed=seed + 2202,
            max_len=max_len
        )

        tr_ids, tr_segs, tr_tags = tag_train
        te_ids, te_segs, te_tags = tag_test

        if len(tr_ids) == 0 or len(te_ids) == 0:
            raise ValueError("Tagging dataset is empty.")

        tag_head = MLPHead(
            d,
            5,
            seed=seed + 2301
        )

        hp = tag_head.parameters()
        hs = snapshot(hp)

        tag_loss_fn = lambda layers, tags: tag_loss(
            tag_head,
            layers[-1],
            tags
        )

        heads["tag"] = run_head(
            "tag",
            hp,
            hs,
            tag_loss_fn,
            (tr_ids, tr_segs, tr_tags),
            lambda: tag_f1(
                params,
                tag_head,
                te_ids,
                te_segs,
                te_tags
            ),
            head_seed=2401
        )

        book_sentence = tag_text(
            params,
            tag_head,
            vocab,
            "mrs green spoke today in new york ."
        )

    seconds["tag"] = time.perf_counter() - block_start

    # =====================================================
    # 13. TASK D — QUESTION ANSWERING
    # =====================================================

    block_start = time.perf_counter()

    qa_n = maximum_budget("qa")

    if qa_n > 0:

        qa_train_pool, qa_train_filler = qa_pools(
            train_stories,
            max_tok=12
        )

        qa_test_pool, qa_test_filler = qa_pools(
            test_stories,
            max_tok=12
        )

        qa_train = make_qa(
            qa_train_pool,
            qa_train_filler,
            vocab,
            qa_n,
            seed=seed + 2501,
            max_len=max_len
        )

        qa_test = make_qa(
            qa_test_pool,
            qa_test_filler,
            vocab,
            evaluation_size("qa"),
            seed=seed + 2502,
            max_len=max_len
        )

        tr_ids, tr_segs, tr_start, tr_end = qa_train
        te_ids, te_segs, te_start, te_end = qa_test

        tr_mask = passage_mask(tr_ids, tr_segs)
        te_mask = passage_mask(te_ids, te_segs)

        qa_head_s = MLPHead(
            d,
            1,
            seed=seed + 2601
        )

        qa_head_e = MLPHead(
            d,
            1,
            seed=seed + 2602
        )

        hp = (
            qa_head_s.parameters()
            + qa_head_e.parameters()
        )

        hs = snapshot(hp)

        qa_loss_fn = lambda layers, start, end, mask: qa_loss(
            qa_head_s,
            qa_head_e,
            layers[-1],
            start,
            end,
            mask
        )

        qa_train_arrays = (
            tr_ids,
            tr_segs,
            tr_start,
            tr_end,
            tr_mask.astype(np.bool_)
        )

        heads["qa"] = run_head(
            "qa",
            hp,
            hs,
            qa_loss_fn,
            qa_train_arrays,
            lambda: qa_accuracy(
                params,
                qa_head_s,
                qa_head_e,
                te_ids,
                te_segs,
                te_start,
                te_end
            ),
            head_seed=2701
        )

    seconds["qa"] = time.perf_counter() - block_start

    # =====================================================
    # 14. RETURN COMPLETE EXPERIMENTAL REPORT
    # =====================================================

    return {
        "vocab_size": actual_vocab_size,

        "history": history,

        "metrics": metrics,

        "future": future,

        "cloze": cloze,

        "neighbours": neighbours,

        "probes": probes,

        "heads": heads,

        "nsp_zero_shot": float(nsp_zero_shot),

        "book_sentence": book_sentence,

        "seconds": seconds
    }

