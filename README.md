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
- [x] **19.** cloze_topk
- [x] **20.** scalar_mix_probe
- [x] **21.** layer_probes
- [x] **22.** MLPHead
- [x] **23.** FineTuner
- [x] **24.** mention_set
- [x] **25.** both_person_set
- [x] **26.** tag_set
- [x] **27.** viterbi_bio
- [x] **28.** make_qa
- [x] **29.** best_span
- [x] **30.** tiny_bert_report

## Results

```
1. Pretraining on unlabeled stories (eqs 15.71-15.73)
   masked loss 7.15 -> 3.43, next-sentence loss 0.711 -> 0.618, 1200 words
   cloze top-1, held-out stories                0.342   [mean 0.38, 0.36 to 0.39]
   cloze top-5                                  0.547   [mean 0.58, 0.56 to 0.60]
   always the most common word                  0.061   [mean 0.07, 0.06 to 0.08]
   next-sentence accuracy (chance 0.5)          0.600   [mean 0.60, 0.58 to 0.61]

2. What the right-hand context is worth (section 15.7)
   cloze with the whole sentence                0.230   [mean 0.33, 0.29 to 0.34]
   cloze with everything after the blank removed 0.110   [mean 0.18, 0.16 to 0.20]

3. Fill in the blank, and what the embeddings learned
   Let's make [MASK] chicken! [SEP] It [MASK] great with orange sauce.  (book: some, tastes)
     blank 1: ['a', 'the', 'to', 'and', 'so']
     blank 2: ['was', 'is', 'had', 'saw', 'felt']
   the little girl went to the [MASK] to play with her [MASK] .
     blank 1: ['loved', 'park', 'play', 'friends', 'new']
     blank 2: ['friends', 'toys', 'play', 'outside', 'mom']
   nearest to sad    ['so', 'happy', 'scared', 'years', 'very', 'walking']
   nearest to happy  ['proud', 'excited', 'scared', 'sad', 'glad', 'shiny']
   nearest to dog    ['neighbor', 'bird', 'tree', 'stick', 'daisy', 'mouse']
   nearest to cake   ['toy', 'dream', 'twins', 'jealous', 'monkey', 'tasty']
   nearest to ran    ['took', 'got', 'turned', 'tried', 'stopped', 'decided']
   sad and happy fill the same slots, so they sit close: masked prediction learns what kind of word a word is,
   and it cannot tell antonyms apart. Sentiment did not transfer reliably, which is why no head here uses it.

4. An ELMo-style scalar mix (section 15.7.1.1): weights over [embedding, layer 1, 2, 3]
   closed-class tagging
     mix weights       [0.063, 0.583, 0.223, 0.131]   [mean [0.06, 0.75, 0.12, 0.08]]
     each layer alone  [0.952, 0.979, 0.96, 0.94]   [mean [0.96, 0.98, 0.95, 0.92]]
   word sense (name or plain word)
     mix weights       [0.08, 0.122, 0.153, 0.644]   [mean [0.05, 0.05, 0.43, 0.47]]
     each layer alone  [0.673, 0.733, 0.854, 0.912]   [mean [0.65, 0.74, 0.93, 0.92]]
   Over five seeds the syntactic probe peaks at layer 1 every time and the sense probe at layer 2 or 3 every time, as section 15.7.1.1 predicts.

5. Four heads on held-out data (scratch: random encoder, frozen: pretrained encoder not trained, fine: fine-tuned)
   Fig 15.35a sentence: which entities does it name? 128 labels, 4 classes
     this run: scratch 0.510   fine 0.746
     5 seeds:  scratch 0.50 (0.47-0.54)   frozen 0.48 (0.45-0.50)   fine 0.72 (0.65-0.80)
   Fig 15.35a sentence: animal, food or thing, on unseen words? 128 labels, 3 classes
     this run: scratch 0.445   fine 0.455
     5 seeds:  scratch 0.45 (0.40-0.51)   frozen 0.44 (0.41-0.47)   fine 0.50 (0.43-0.57)
   Fig 15.35b pair: do both sentences name a person? 256 labels
     this run: scratch 0.672   fine 0.775
     5 seeds:  scratch 0.65 (0.62-0.68)   frozen 0.69 (0.65-0.75)   fine 0.77 (0.73-0.79)
   Fig 15.35b pair: are these from the same story? no labels at all, the pretrained next-sentence head as it stands
     this run: 0.535
     5 seeds:  mean 0.58, 0.55 to 0.61
     for comparison, trained on 256 labels, 5 seeds: scratch 0.54 (0.52-0.56)   frozen 0.57 (0.54-0.60)   fine 0.56 (0.53-0.58)
   Fig 15.35c tagging: entity-level F1 after Viterbi decoding, 128 labels
     this run: scratch 0.078   frozen 0.357   fine 0.559
     5 seeds:  scratch 0.12 (0.05-0.21)   frozen 0.50 (0.41-0.58)   fine 0.76 (0.67-0.84)
   Fig 15.35d question answering: exact span, 128 labels
     this run: scratch 0.373   frozen 0.307   fine 0.457
     5 seeds:  scratch 0.37 (0.33-0.41)   frozen 0.47 (0.42-0.50)   fine 0.70 (0.58-0.82)
   A frozen encoder finds entities but cannot use the question, so it stays below the 0.5 you get by ignoring the question.

6. The first half of the book's sentence, tagged by the fine-tuned model
   mrs/B-Per  green/I-Per  spoke/O  today/O  in/O  new/O  york/B-Loc  ./O

seconds at the end of each block: data 35, pretrain 24, measurements 183, probes 40, mention 18, topic 20, both_person 17, same_story 0, tag 26, qa 30
study finished in 395 s
```
