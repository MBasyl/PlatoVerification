# Original script from https://github.com/mikekestemont/ruzicka
# Adapted for Python3.9+ from https://github.com/bnagy/ruzicka/blob/main/README.md

### Install:
# pip install git+https://github.com/bnagy/ruzicka@main#egg=ruzicka

from ruzicka.Order2Verifier import Order2Verifier, Order1Verifier
from sklearn.preprocessing import LabelEncoder
from ruzicka.utilities import *
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

##############
# set Parameters
features = ['char_wb']  # 'char/word' are
ngrams = [3, 4, 5]  # 3, 4,5
bases = ['profile', 'instance']
vectors = ['tf_idf', 'tf_std']
metric = 'minmax'
nb_bootstrap_iter = 100
rnd_prop = 0.5
nb_imposters = 30
mfi = 500  # number of most frequent features
min_df = 2  # culling

##############
# get imposter data:
train_data, _ = load_pan_dataset('data/latin/dev')
train_labels, train_documents = zip(*train_data)

# get test data:
test_data, _ = load_pan_dataset('data/latin/test')
test_labels, test_documents = zip(*test_data)

# fit encoder for author labels:
label_encoder = LabelEncoder()
label_encoder.fit(train_labels+test_labels)
train_ints = label_encoder.transform(train_labels)
test_ints = label_encoder.transform(test_labels)

for base in bases:
    for vector in vectors:
        for feature in features:
            for ngram in ngrams:
                print("::::::::::::")
                print("base:", base, "\tvector:", vector,
                      "\tfeature:", feature, "\tngram:", ngram)
                print("::::::::::::\n\n")

                # fit vectorizer:
                vectorizer = Vectorizer(mfi=mfi,
                                        vector_space=vector,
                                        ngram_type=feature,
                                        ngram_size=ngram)
                vectorizer.fit(train_documents+test_documents)
                train_X = vectorizer.transform(train_documents)  # .toarray()
                test_X = vectorizer.transform(test_documents)  # .toarray()
                # exit(0)
                cols = ['label']
                for test_author in sorted(set(test_ints)):
                    auth_label = label_encoder.inverse_transform([test_author])[
                        0]
                    cols.append(auth_label)

                proba_df = pd.DataFrame(columns=cols)

                for idx in range(len(test_documents)):
                    target_auth = test_ints[idx]
                    target_docu = test_X[idx]
                    non_target_test_ints = np.array(
                        [test_ints[i] for i in range(len(test_ints)) if i != idx])
                    non_target_test_X = np.array([test_X[i]
                                                  for i in range(len(test_ints)) if i != idx])
                    tmp_train_X = np.vstack((train_X, non_target_test_X))
                    tmp_train_y = np.hstack((train_ints, non_target_test_ints))

                    tmp_test_X, tmp_test_y = [], []
                    for t_auth in sorted(set(test_ints)):
                        tmp_test_X.append(target_docu)
                        tmp_test_y.append(t_auth)

                    # fit the verifier:
                    verifier = Order2Verifier(metric=metric,
                                              base=base,
                                              nb_bootstrap_iter=nb_bootstrap_iter,
                                              rnd_prop=rnd_prop
                                              )
                    verifier.fit(tmp_train_X, tmp_train_y)
                    probas = verifier.predict_proba(test_X=tmp_test_X,
                                                    test_y=tmp_test_y,
                                                    nb_imposters=nb_imposters
                                                    )

                    row = [label_encoder.inverse_transform(
                        [target_auth])[0]]  # author label
                    row += list(probas)
                    # print(row)
                    proba_df.loc[len(proba_df)] = row

                proba_df = proba_df.set_index('label')

                # write away score tables:
                proba_df.to_csv('output/tab'+base+'_'+feature +
                                '_'+str(ngram)+'_'+metric+'_'+vector+'.csv')
