# Pretrain a Tiny BERT, Then Fine-Tune Four Heads

Murphy's Probabilistic Machine Learning makes a promise: fit a language model on a large unlabeled corpus, then fine-tune it on a small labeled set. This project tests the promise on a CPU, in tinygrad, with everything written by you. You plant people and places into TinyStories so labels exist by construction (the book's "Mrs Green" and "New York", with surnames that are also ordinary words), split the names so the test only uses ones the labels never showed, and build a three-layer bidirectional encoder from the book's pseudocode. You pretrain it with the masked language model (tied embeddings, fixed mask slots) and next-sentence prediction, read its cloze guesses on the book's own example, and measure what the words after the blank are worth. An ELMo-style scalar mix then tests the book's claim that syntactic tasks favour lower layers and semantic ones higher layers. Finally you add the four heads of Fig 15.35: sentence and pair classification from [CLS], BIO tagging with a constrained Viterbi decoder and entity-level F1, and question answering with start and end MLPs. Each head runs from random weights, from a frozen pretrained encoder and from a fine-tuned one, on the same few labels. Headline numbers, mean over five seeds on the prod CPU image, test data with held-out names, places and words, in the order scratch, frozen, fine-tuned. Tagging F1 on 128 labels: 0.12, 0.50, 0.76. Question answering exact-span accuracy on 128 labels: 0.37, 0.47, 0.70. Which entities a sentence names, on 128 labels: 0.50, 0.48, 0.72. Whether both sentences name a person, on 256 labels: 0.65, 0.69, 0.77. Pretraining reaches cloze top-1 0.38 and top-5 0.58 against 0.07 for the most common word, and next-sentence accuracy 0.60; removing the words after the blank drops cloze from 0.33 to 0.18. The book's layer claim reproduces on all five seeds: the syntactic probe puts 0.75 of its weight on layer 1, the sense probe about 0.9 on layers 2 and 3. Some things do not work at this scale, and the project says so. Sentiment does not transfer reliably, since masked prediction places sad next to happy. Topic classification on unseen words gains only about 5 points at 128 labels. Same-story classification stays at 0.54 to 0.56, though the next-sentence head with no labels at all gets 0.58. A question that names one of two people stayed under 50 percent even on 2,000 labels. The assembled script runs one seed of the whole study in five to seven minutes, depending on how busy the host is.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** split_sentences
- [x] **2.** load_stories
- [x] **3.** entity_lists
- [x] **4.** plant_entities
- [x] **5.** build_vocab
- [x] **6.** pack_pair
- [x] **7.** init_params
- [x] **8.** embed_inputs
- [x] **9.** self_attention
- [x] **10.** encoder_block
- [x] **11.** encode
- [x] **12.** mask_tokens
- [x] **13.** mlm_loss
- [x] **14.** sample_nsp_batch
- [x] **15.** nsp_loss
- [x] **16.** Adam
- [x] **17.** pretrain
- [x] **18.** mlm_metrics

---

Built on Deep-ML.
