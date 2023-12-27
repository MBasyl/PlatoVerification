# Original script from https://github.com/mikekestemont/ruzicka
# Adapted for Python3.9+ from https://github.com/bnagy/ruzicka/blob/main/README.md

# Install:
# pip install git+https://github.com/bnagy/ruzicka@main#egg=ruzicka

from ruzicka.Order2Verifier import Order2Verifier
# from ruzicka.Order1Verifier import Order1Verifier
from sklearn.preprocessing import LabelEncoder
from ruzicka.utilities import *
from ruzicka.score_shifting import ScoreShifter

import pandas as pd
import numpy as np
from ruzicka.plot_res import plot_heatmap

sys.stdout = open(f'GI_log.txt', 'a')


def load_dataset(directory, max_number_samples=None, ext="txt", encoding="utf8"):
    """
    Loads the data from `directory`, which should hold subdirs
    for each "problem"/author in a dataset. 

    Parameters
    ----------
    directory: str, default=None
        Path the directory from which the `problems` are loaded.
    ext: str, default='txt'
        Only loads files with this extension,
        useful to filter out e.g. OS-files.

    Returns
    ----------
    labels:
        list of author names
    data:
        list of documents

    """
    data = []

    for author in sorted(os.listdir(directory)):
        # print(author)
        path = os.sep.join((directory, author))
        if os.path.isdir(path):
            for filepath in sorted(glob.glob(path + "/*." + ext)):
                text = codecs.open(filepath, mode="r").read()
                file = os.path.splitext(os.path.basename(filepath))[0]
                number = file.split("_")[1]
                if max_number_samples:
                    if int(number) < max_number_samples:
                        # combine same work?
                        data.append((author, text))
                else:
                    data.append((author, text))

    return data


def main():

    # fit encoder for author labels:
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels+test_labels)
    train_ints = label_encoder.transform(train_labels)
    test_ints = label_encoder.transform(test_labels)

    vectorizer = Vectorizer(mfi=mfi,
                            vector_space=vector,
                            ngram_type=feature,
                            ngram_size=ngram)
    verifier = Order2Verifier(metric=metric,
                              base=base,
                              nb_bootstrap_iter=nb_bootstrap_iter,
                              rnd_prop=rnd_prop
                              )
    # get benchmark
    splitter = splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    shifter = ScoreShifter()
    shifter = fit_shifter(np.array(train_documents), train_ints,
                          vectorizer, verifier, shifter, test_size=0.2)
    benchmark_imposters(np.array(train_documents), train_ints,
                        splitter, vectorizer, verifier, shifter)
    # fit vectorizer:
    train_X = vectorizer.fit_transform(
        train_documents)  # .toarray()
    test_X = vectorizer.transform(test_documents)  # .toarray()

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

        predicted = verifier.fit(tmp_train_X, tmp_train_y)
        print(predicted)
        probas = verifier.predict_proba(test_X=tmp_test_X,
                                        test_y=tmp_test_y,
                                        nb_imposters=nb_imposters
                                        )

        row = [label_encoder.inverse_transform(
            [target_auth])[0]]  # author label
        row += list(probas)  # probas!!
        # print(row)
        proba_df.loc[len(proba_df)] = row

    proba_df = proba_df.set_index('label')

    return proba_df


if __name__ == "__main__":

    # get imposter data:
    train_data = load_dataset('data/GIprofiles/train')
    train_labels, train_documents = zip(*train_data)

    # get test data:
    test_data = load_dataset('data/GIprofiles/test')  # 5
    test_labels, test_documents = zip(*test_data)

    print("train documents:", len(train_documents), "labels:", len(train_labels))
    print("test documents:", len(test_documents), "labels:", len(test_labels))

    ##############
    # set Parameters
    feature = 'char_wb'  # 'char/word' char_wb
    ngram = 5  # 3, 4,5 # ALSO TRY RANGE (3,5)
    base = 'profile'  # , 'instance'
    vector = 'tf_idf'  # 'tf_std',
    metric = 'minmax'
    nb_bootstrap_iter = 100
    rnd_prop = 0.5  # aka number of feature selected per iteration
    nb_imposters = 10  # aka 90%?
    mfi = 500  # number of most frequent features
    min_df = 4  # culling

    #############
    # Hip, XenApo, Epi, Parm

    filename = 'Plato_Instances_plain_c5.csv'
    print(filename)

    df_res = main()
    # write away score tables:
    df_res.to_csv(filename)

    plot_heatmap(filename)
