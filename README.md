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

## Preprocessing (main.py)

1. Extract data from TEI to text (processXML.py)
2. create NER list (NERtokens.py) to inspect manually
3. Use getTXTstatistics.py to obtain general overview of the corpus

- remove texts with less than 2.000 words
- remove 1200 words from Xenophon;9800 from Aristoteles, carefully reduce Plato to 300.000 by balancing early-mid-late-laws works.

4. create POS-tagged dataset (processPOS.py)
5. clean text from extra spaces and punctuation

- NB: capitalize after periods and keep diacritics

6. substitute NER with "\*" to reduce topical bias
7. Setup directories for ML

# Experimental Set-up

## Unsupervised

- Clustering performed on full author profiles and on single documents
- PCA?

## Supervised

Test-run: On polygraphous Xenophon (tlg: 0032): 14 texts

- classifiers
  - SVM using code adapted from https://github.com/avjves/AuthAttHelper/blob/master/wrapper.py#L73
    #build PIPELINE FOR IT!!! C_CHOICE > CV > FIRST PLOT > MAIN > BEST_FEAT > LAST PLOT
  - GI-profile and instance using code adapted from https://github.com/mikekestemont/ruzicka
    #use parse_dir for data setup > parformGI.py choose parameters
  - Masking?
  - Diff-Vec?
- Neural Networks
  - LSTM
