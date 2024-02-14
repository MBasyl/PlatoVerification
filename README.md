
> [!WARNING]
> ***WORK IN PROGRESS....***

# Data

## collection

From [PerseusDL](https://github.com/PerseusDL/canonical-greekLit) and [OpenGreekAndLatin](https://github.com/OpenGreekAndLatin/First1KGreek).

- Plato (tlg0059), 36 works we split in two corpora: Plato and pseudoPlato (including the VII Epistle)
- Xenophon (tlg0032), 14 works

## STEP 1 : PREPROCESSING 
Use `main.py` to extract data from TEI to clean TXT and obtain two datasets: **PLAIN** text and **PARSED** part-of-speech masked.
All steps hereafter need to be performed separately for the two datasets.

Use `makedfprofiles.py` to create a DataFrame. Concatenates files into chronological profiles (Plato_Early, Plato_Mature, Plato_Late) and genre profiles (Xen_Dialogues, Xen_Treatise, Xen_Histories) and chunks the six documents plus Pseudo-Platonic dialogues into 5.000 word instances. Adds binary labels that identify Plato_Late as the positive class.

## STEP 2 : UNSUPERVISED STUDY
Perform Cluster Analysis using the [Stylo package](https://pages.github.com/)in R (``) 
Perform Support Vector Data Description (`performSVDD.py`, adapted from [Kepeng Qiu](https://github.com/iqiukp/SVDD-Python)'s GitHub) 

## STEP 3: SUPERVISED STUDY SETUP
- Revise files assigned to positive label based on unsupervised insights
- Create ad-hoc testset. Use `obfuscate.py` to obfuscate 5\% of Plato's _Laws_ with different percentages (50-90\%) of Pseudo_Plato _Lovers_. This will allow a more fine-grained comprehension of the threshold for performance metrics
- Remove from dataset 1. Lovers 2. The chunks of authentic Laws used in obfuscation
**Models compared in this study**: 
- Support Vector Machine (SVM)
- Common N-Gram (CNG) (adapted from [Robert Layton](https://github.com/robertlayton/authorship_tutorials)'s tutorials)

## STEP 4: SUPERVISED STUDY
### SVM model (`performSVM.py`)
- Use SearchGrid to find best parameters for: 
	- TF-IDF vectorizer: {'max_features': [1000, 3000, 5000], 'max_df': [1, 0.4, 0.8], 'ngram_range': [(4, 4), (5,5), (6,6), (4,6)]}
	- Classifier: {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 100], 'class_weight':[None, 'balanced' ]}
- Use 6-fold cross-validation to obtain the mean F1 and AUC-ROC
- Execute SVM model on train-test 
- Get insights into learned features and predictions (``) with [SHAP package](https://pages.github.com/)
- Evaluate model by training the whole dataset and testing on Validation set


### CNG-RLP model (``) 

