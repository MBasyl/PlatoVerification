
# Author: Arman Naseri Jahfari (a.naserijahfari@tudelft.nl)

import numpy as np
from SVDD import SVDD
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pandas as pd

# SMALLER BANDWITH BETTER! but more overfitting
# OBSERVATIONS for PARSED
# LL meglio di LLM
# small ngram difference (4,5) meglio di large (3,6)
# PC works, AC too confused
# lims=[-1, 2, -1, 1]
# OBERSVARIONS FOR PLAIN
# bigger ngram diff (4-8) meglio di small (4-6)
# best lims =[-0.5, 1.5, -0.5, 0.5]


def vectorize_data(X_train, X_test, feature, ngram):

    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                            analyzer=feature, ngram_range=ngram, max_features=3000)

    # Training data
    X_train_tfidf = tfidf.fit_transform(X_train)
    # Test data
    X_test_tfidf = tfidf.transform(X_test)

    scaler = TruncatedSVD(n_components=2)
    X_train = scaler.fit_transform(X_train_tfidf)
    X_test = scaler.transform(X_test_tfidf)

    return X_train, X_test


title = 'PlainPC_Lc36_SVDDband03'
print("\n\n", title)
#  training data contains outliers which are defined as observations that are far from the others.
f = "data/benchmarkData/PLAINdata.csv"
df = pd.read_csv(f)
fother = "data/benchmarkData/PLAINDataotherAuthors.csv"
df2 = pd.read_csv(fother)
# df = pd.concat([df1, df2], ignore_index=True, axis=0)

df['binary_label'] = df['label'].apply(
    lambda x: 1 if x in [7] else -1)

testdata = df[df['author'].str.contains('test')]
X_test = testdata['text']
ytest = testdata['binary_label']
test_names = testdata['author'].to_list()

traindata = df[~df['author'].str.contains('test')]
train_names = traindata['author'].to_list()
X_train = traindata['text']
ytrain = traindata['binary_label']

Xtrain, Xtest = vectorize_data(X_train, X_test, feature='char', ngram=(3, 6))


clf = SVDD(kernel_type='rbf', bandwidth=0.3)  # , fracrej=np.array([0.5, 0.1]))
clf.fit(Xtrain, ytrain)
y_pred = clf.predict(Xtest)

print(confusion_matrix(ytest, y_pred, normalize='true'))
p = clf._plot_contour(Xtrain, ytrain, document_names=train_names,
                      lims=[-0.5, 1.5, -0.5, 0.5])
p.savefig(f'{title}_train.png', bbox_inches='tight', pad_inches=0)
d = clf._plot_contour(Xtest, ytest, document_names=test_names,
                      lims=[-0.5, 1.5, -0.5, 0.5])
d.savefig(f'{title}_test.png', bbox_inches='tight', pad_inches=0)
