# WORK IN PROGRESS....

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
- Philosophers
  - Aristoteles (tlg: 0086)
  - Xenocrates of Chalcedon (tlg: 0634)

## Preprocessing (`main.py`)

1. Extract data from TEI to text (`processXML.py`)
2. Use getTXTstatistics.py to obtain general overview of the corpus

- remove texts with less than 2.000 words
  #remove 1200 words from Xenophon;9800 from Aristoteles, carefully reduce Plato to 300.000 by balancing early-mid-late-laws works.

3. create NER list (`NERtokens.py`) to inspect manually
4. clean text from extra spaces and punctuation

- NB: capitalize after periods and keep diacritics

# Experimental Set-up

## Unsupervised

- Clustering performed on full author profiles and on single documents
- Use Stylo [REF]

## Supervised

5. Set up directories for ML: PARSED and PLAIN
6. Datset for supervised learning: Late_Plato profile + rest of dataset exluding PsPla_Mex and PsPla_VIII
7. create POS-tagged dataset (`processPOS.py` on PARSED dir)
8. substitute NER with "\*" to reduce topical bias (use `replace_named_entities()` on PLAIN dir)
