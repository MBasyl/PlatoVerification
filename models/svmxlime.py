# Looks into WORD UNIGRAMS with cool visuals
"""@inproceedings{lime,
  author    = {Marco Tulio Ribeiro and
               Sameer Singh and
               Carlos Guestrin},
  title     = {"Why Should {I} Trust You?": Explaining the Predictions of Any Classifier},
  booktitle = {Proceedings of the 22nd {ACM} {SIGKDD} International Conference on
               Knowledge Discovery and Data Mining, San Francisco, CA, USA, August
               13-17, 2016},
  pages     = {1135--1144},
  year      = {2016},
}"""

from __future__ import print_function
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

data = 'PLAIN'
feature = 'word'
csv_data = f"{data}dataset_obfuscate.csv"

df = pd.read_csv(csv_data)
val_set = df[df['title'].str.contains('VII|#')]

# Train-test split, keeping class imbalance
df = df[~df['title'].str.contains('VII|#')]
X = df['text']
y = df['label']
class_names = ['Not-Plato', 'Plato']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3, stratify=y)

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                        analyzer='word',
                        max_features=1000, max_df=0.8)
scaler = StandardScaler(with_mean=False)

model = SVC(kernel='linear', C=100, probability=True, random_state=54)
c = make_pipeline(tfidf, scaler, model)

# Train the model
c.fit(X_train, y_train)
# feature_names = c[:-1].get_feature_names_out()
# proba = c.predict_proba([X_test[0]])  # as list

explainer = LimeTextExplainer(class_names=class_names, bow=False)
idx = 22  # 30Pla  98, 22, 91, 73

exp = explainer.explain_instance(
    X_test[idx], c.predict_proba, num_features=20)
print('Document id: %d' % idx)
print('Probability of being Plato =',
      c.predict_proba([X_test[idx]])[0, 1])
print('True class: %s' % class_names[y_test[idx]])
print(exp.as_list())
exp.save_to_file('limeXAIbigrams.html')
fig = exp.as_pyplot_figure()
fig.savefig("limebar.png")
