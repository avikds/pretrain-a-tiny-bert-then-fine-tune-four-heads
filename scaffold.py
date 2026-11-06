"""
Pretrain a Tiny BERT, Then Fine-Tune Four Heads scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Pretrain a Tiny BERT, Then Fine-Tune Four Heads.

One run of the study on the CPU image: pretrain on TinyStories with the masked language model and
next-sentence losses, measure it, probe its layers, then fine-tune the four heads of Fig 15.35 on a
few labels and test them on held-out names, places and words. The bracketed ranges are the same
numbers over five seeds (0-4) measured on this image when the project was calibrated.
"""
import time

RANGES = {'top1': 'mean 0.38, 0.36 to 0.39',
 'top5': 'mean 0.58, 0.56 to 0.60',
 'most_common': 'mean 0.07, 0.06 to 0.08',
 'nsp': 'mean 0.60, 0.58 to 0.61',
 'future_both': 'mean 0.33, 0.29 to 0.34',
 'future_left': 'mean 0.18, 0.16 to 0.20',
 'w_syntactic': 'mean [0.06, 0.75, 0.12, 0.08]',
 'a_syntactic': 'mean [0.96, 0.98, 0.95, 0.92]',
 'w_sense': 'mean [0.05, 0.05, 0.43, 0.47]',
 'a_sense': 'mean [0.65, 0.74, 0.93, 0.92]',
 'layers_note': 'Over five seeds the syntactic probe peaks at layer 1 every time and the sense probe at layer 2 or 3 '
                'every time, as section 15.7.1.1 predicts.',
 'mention_128': 'scratch 0.50 (0.47-0.54)   frozen 0.48 (0.45-0.50)   fine 0.72 (0.65-0.80)',
 'topic_128': 'scratch 0.45 (0.40-0.51)   frozen 0.44 (0.41-0.47)   fine 0.50 (0.43-0.57)',
 'both_person_256': 'scratch 0.65 (0.62-0.68)   frozen 0.69 (0.65-0.75)   fine 0.77 (0.73-0.79)',
 'same_story_256': 'scratch 0.54 (0.52-0.56)   frozen 0.57 (0.54-0.60)   fine 0.56 (0.53-0.58)',
 'tag_128': 'scratch 0.12 (0.05-0.21)   frozen 0.50 (0.41-0.58)   fine 0.76 (0.67-0.84)',
 'qa_128': 'scratch 0.37 (0.33-0.41)   frozen 0.47 (0.42-0.50)   fine 0.70 (0.58-0.82)',
 'story_zero': 'mean 0.58, 0.55 to 0.61'}


def show(key):
    return f"[{RANGES[key]}]"


def line(label, value, key):
    print(f"   {label:44s} {value:.3f}   {show(key)}")


def main() -> None:
    t0 = time.time()
    rep = tiny_bert_report(seed=0)
    m = rep["metrics"]
    h = rep["history"]

    print("1. Pretraining on unlabeled stories (eqs 15.71-15.73)")
    print(f"   masked loss {h[0]['mlm']:.2f} -> {h[-1]['mlm']:.2f}, next-sentence loss {h[0]['nsp']:.3f} -> {h[-1]['nsp']:.3f}, {rep['vocab_size']} words")
    line("cloze top-1, held-out stories", m["top1"], "top1")
    line("cloze top-5", m["top5"], "top5")
    line("always the most common word", m["most_common"], "most_common")
    line("next-sentence accuracy (chance 0.5)", m["nsp"], "nsp")

    print("\n2. What the right-hand context is worth (section 15.7)")
    both, left = rep["future"]
    line("cloze with the whole sentence", both, "future_both")
    line("cloze with everything after the blank removed", left, "future_left")

    print("\n3. Fill in the blank, and what the embeddings learned")
    print("   Let's make [MASK] chicken! [SEP] It [MASK] great with orange sauce.  (book: some, tastes)")
    for j, guesses in enumerate(rep["cloze"]["book"]):
        print(f"     blank {j + 1}: {guesses}")
    print("   the little girl went to the [MASK] to play with her [MASK] .")
    for j, guesses in enumerate(rep["cloze"]["story"]):
        print(f"     blank {j + 1}: {guesses}")
    for w, near in rep["neighbours"].items():
        print(f"   nearest to {w:6s} {near}")
    print("   sad and happy fill the same slots, so they sit close: masked prediction learns what kind of word a word is,")
    print("   and it cannot tell antonyms apart. Sentiment did not transfer reliably, which is why no head here uses it.")

    print("\n4. An ELMo-style scalar mix (section 15.7.1.1): weights over [embedding, layer 1, 2, 3]")
    for task, label in (("syntactic", "closed-class tagging"), ("sense", "word sense (name or plain word)")):
        r = rep["probes"][task]
        print(f"   {label}")
        print(f"     mix weights       {[round(x, 3) for x in r['weights']]}   {show('w_' + task)}")
        print(f"     each layer alone  {[round(x, 3) for x in r['layer_acc']]}   {show('a_' + task)}")
    print(f"   {RANGES['layers_note']}")

    print("\n5. Four heads on held-out data (scratch: random encoder, frozen: pretrained encoder not trained, fine: fine-tuned)")
    heads = rep["heads"]

    def row(task, n, label, f1=False):
        cells = []
        for cond in ("scratch", "frozen", "fine"):
            if cond in heads[task][n]:
                v = heads[task][n][cond]
                cells.append(f"{cond} {(v[0] if f1 else v):.3f}")
        print(f"   {label}")
        print(f"     this run: {'   '.join(cells)}")
        print(f"     5 seeds:  {RANGES[task + '_' + str(n)]}")

    row("mention", 128, "Fig 15.35a sentence: which entities does it name? 128 labels, 4 classes")
    row("topic", 128, "Fig 15.35a sentence: animal, food or thing, on unseen words? 128 labels, 3 classes")
    row("both_person", 256, "Fig 15.35b pair: do both sentences name a person? 256 labels")
    print("   Fig 15.35b pair: are these from the same story? no labels at all, the pretrained next-sentence head as it stands")
    print(f"     this run: {rep['nsp_zero_shot']:.3f}")
    print(f"     5 seeds:  {RANGES['story_zero']}")
    print(f"     for comparison, trained on 256 labels, 5 seeds: {RANGES['same_story_256']}")
    row("tag", 128, "Fig 15.35c tagging: entity-level F1 after Viterbi decoding, 128 labels", f1=True)
    row("qa", 128, "Fig 15.35d question answering: exact span, 128 labels")
    print("   A frozen encoder finds entities but cannot use the question, so it stays below the 0.5 you get by ignoring the question.")

    print("\n6. The first half of the book's sentence, tagged by the fine-tuned model")
    print("   " + "  ".join(f"{w}/{t}" for w, t in rep["book_sentence"]))

    print("\nseconds at the end of each block: " + ", ".join(f"{k} {v:.0f}" for k, v in rep["seconds"].items()))
    print(f"study finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

