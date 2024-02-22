> [!WARNING] > **_WORK IN PROGRESS...._**

<!-- https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax -->

# Data

## collection

From [PerseusDL](https://github.com/PerseusDL/canonical-greekLit) and [OpenGreekAndLatin](https://github.com/OpenGreekAndLatin/First1KGreek).

- Plato (tlg0059), 36 works we split in two corpora: Plato and pseudoPlato (including the VII Epistle);
- Xenophon (tlg0032), 14 works.

## STEP 1 : PREPROCESSING

Use `main.py` to extract data from TEI to clean TXT and obtain two datasets: **PLAIN** text and **PARSED** part-of-speech masked.
All steps hereafter need to be performed separately for the two datasets.

Use `makedfprofiles.py` to create a DataFrame. Concatenates files into chronological profiles (Plato_Early, Plato_Mature, Plato_Late) and genre profiles (Xen_Dialogues, Xen_Treatise, Xen_Histories) and chunks the six documents plus Pseudo-Platonic dialogues into 5.000 word instances. Adds binary labels that identify Plato_Late as the positive class.

## STEP 2 : UNSUPERVISED STUDY

Perform **Cluster Analysis** using the [Stylo package](https://github.com/computationalstylistics/stylo)in R (``).[[1]](#1) 
\\
Perform **Support Vector Data Description** (`performSVDD.py`, adapted from [Kepeng Qiu](https://github.com/iqiukp/SVDD-Python)'s GitHub).

## STEP 3: SUPERVISED STUDY

- Revise files assigned to positive label based on unsupervised insights
- Create ad-hoc testset. Use `obfuscate.py` to obfuscate 5\% of Plato's _Laws_ with different percentages (50-90\%) of Pseudo-Plato _Lovers_. This will allow a more fine-grained comprehension of the threshold for performance metrics.
- Remove from dataset: _Lovers_ and the chunks of authentic _Laws_ used in obfuscation.
- Compare performace of a **Support Vector Machine** (SVM) Machine Learning model and a **Common N-Gram**(CNG) model.

### SVM model (`performSVM.py`)

- Use SearchGrid to find best parameters for:
  - TF-IDF vectorizer: {'max_features': [1000, 3000, 5000], 'max_df': [1, 0.4, 0.8], 'ngram_range': [(4, 4), (5,5), (6,6), (4,6)]}
  - Classifier: {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 100], 'class_weight':[None, 'balanced' ]}
- Use 6-fold cross-validation to obtain the mean F1 and AUC-ROC
- Execute SVM model on train-test
- Get insights into learned features and predictions (`svmXAI.py`) with [SHAP package](https://github.com/shap/shap?tab=readme-ov-file)[[2]](#2)
- Evaluate model by training the whole dataset and testing on Validation set

### CNG model (`performUnaryCNG.py`)

- Unary classification method. Script adapted from [Robert Layton](https://github.com/robertlayton/authorship_tutorials)'s tutorials [[3]](#3)(cfr.`RLP.py` model) based on the CNG method[[4]](#4).

## References

<a id="1">[1]</a> Eder, M., Rybicki, J. and Kestemont, M. (2016). Stylometry with R: a package for computational text analysis. R Journal, 8(1): 107-21. https://journal.r-project.org/archive/2016/RJ-2016-007/index.html

<a id="2">[2]</a> Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.

<a id="3">[3]</a> Layton, R., Watters, P., & Dazeley, R. (2012). Recentred local profiles for authorship attribution. Natural Language Engineering, 18(3), 293-312.

<a id="4">[4]</a> Kešelj, V., Peng, F., Cercone, N., & Thomas, C. (2003, August). N-gram-based author profiles for authorship attribution. In Proceedings of the conference pacific association for computational linguistics, PACLING (Vol. 3, pp. 255-264).
