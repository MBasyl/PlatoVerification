# Data

## collection

from https://github.com/PerseusDL/canonical-greekLit and https://github.com/OpenGreekAndLatin/First1KGreek

- Plato (tlg: 0059), then divided in:
  - Plato
  - pseudoPlato
  - Epistles
- Coeve Orators (prose)
  - Alcidamas (tlg: 0610)
  - Demosthenes (tlg: 0014)
  - Isocrates (tlg: 0010)
  - Lysias (tlg: 0540)
  - Xenophon (tlg: 0032)
  - Hyperides (tlg: 0030)
  - Gorgias (tlg: 0593)
- Epistolographers
  - Epicurus (tlg: 0537)
- Philosophers
  - Aristoteles (tlg: 0086)
  - Xenocrates of Chalcedon (tlg: 0634)
  - Speusippus fragmenta (tlg: 1692)

## Preprocessing (processAll.py)

- Extracting data from TEI to text
- create NER list to inspect manually
- create corpus stats -> delete texts with less than 2k words
- create lemmatized and POS-tag dataset
- clean text from diacritics(?)
- sub NER with "x" to avoid topical bias

# Experimental Set-up

## Unsupervised

- Clustering
- PCA?

## Supervised

Test-run: On polygraphous Xenophon (tlg: 0032): 14 texts

- classifiers
  - SVM
  - GI
  - Masking
  - Diff-Vec
- Neural Networks
  - CNN
  - LSTM
  - SBERT
