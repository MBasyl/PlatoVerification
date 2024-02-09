# perform and OCSVM
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.lines as mlines
import matplotlib.font_manager
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import OneClassSVM
import pandas as pd
from sklearn.metrics import make_scorer, f1_score, confusion_matrix

# !!!!!!!!!
# OUTLIER DETECTION (SVDD): training data contains outliers which are defined as observations that are far from the others.
# Outlier detection estimators thus try to fit the regions where the training data is the most concentrated,
# ignoring the deviant observations.
# NOVELTY DETECTION (OCSVM): The training data is not polluted by outliers and we are interested in detecting whether a
# new observation is an outlier. In this context an outlier is also called a novelty.
# !!!!!!!!!


def reduce_dimentionality(X_train, X_test):
    # Reduce Dimensionality
    # TSNE(TNSE, init='random', random_state=42)
    svd = TruncatedSVD(n_components=2)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)

    return X_train_svd, X_test_svd


def plot_unsupervised_model(X_train, X_test, title: str, xlim: tuple, ylim: tuple, document_names=None):
    print(f"\nMaking a scatter plot")
    X_train_svd, X_test_svd = reduce_dimentionality(
        X_train, X_test)

    clf = ocsvm.fit(X_train_svd)

    # OCSVM
    _, ax = plt.subplots()

    # generate grid for the boundary display
    num_points = 10  # Adjust this number for density
    x_range = np.linspace(X_train_svd[:, 0].min(),
                          (X_train_svd[:, 0].max()), num_points)
    y_range = np.linspace(X_train_svd[:, 1].min(),
                          X_train_svd[:, 1].max(), num_points)

    xx, yy = np.meshgrid(x_range, y_range)

    X = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)

    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="decision_function",
        plot_method="contour",
        ax=ax,
        levels=[0],
        colors="darkred",
        linewidths=2,
    )

    s = 40
    b1 = ax.scatter(X_train_svd[:, 0], X_train_svd[:, 1], c='skyblue',
                    edgecolor='k', s=s)

    b2 = ax.scatter(X_test_svd[:, 0], X_test_svd[:, 1], c=y_test,
                    cmap='viridis', edgecolor='k', s=s)
    if document_names:
        # Label scattered dots with document names
        for i, name in enumerate(document_names):
            plt.text(X_test_svd[i, 0], X_test_svd[i, 1],
                     name, fontsize=7, ha='right')

    plt.legend(
        [mlines.Line2D([], [], color="darkred"), b1, b2],
        [
            "Decision boundary",
            "Plato documents",
            "Unseen documents",
        ],
        loc="upper right",
        prop=matplotlib.font_manager.FontProperties(size=10),
    )

    min_x, max_x = X_train_svd[:, 0].min(), X_train_svd[:, 0].max()
    min_y, max_y = X_train_svd[:, 1].min(), X_train_svd[:, 1].max()

    ax.set(
        title="One-class Novelty Detection",
        # xlim=(min_x, max_x),
        # ylim=(min_y, max_y),
        xlim=xlim,
        ylim=ylim
    )
    plt.savefig(f'{title}.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def load_data(data: pd.DataFrame, labels: str):
    print("\nLoading data...")
    # turn classes into binary 1/-1 label for one-class SV
    if labels == 'Law':
        data['binary_label'] = data['label'].apply(
            lambda x: 1 if x in [7] else -1)
    elif labels == 'Law|Late':
        data['binary_label'] = data['label'].apply(
            lambda x: 1 if x in [7, 9] else -1)
    elif labels == 'Law|Late|Mature':
        data['binary_label'] = data['label'].apply(
            lambda x: 1 if x in [7, 9, 12] else -1)

    # Split Train-Test
    traindata = data[~data['author'].str.contains(
        'test') & (data['binary_label'] == 1)]
    testdata = data[data['author'].str.contains(
        'test') | (data['binary_label'] != 1)]

    Xtest = testdata['text']
    ytest = testdata['binary_label']
    test_authors = testdata['author'].tolist()

    Xtrain = traindata['text']
    ytrain = traindata['binary_label']

    print("\n\nOverview number authors:", len(set(traindata.author.tolist())))
    print("Train Authors", set(traindata.author.tolist()),
          "\nBalance: ", ytrain.value_counts())
    print("Test Authors", set(testdata.author.tolist()),
          "\nBalance: ", ytest.value_counts())

    return Xtrain, ytrain, Xtest, ytest, test_authors


def vectorize_data(Xtrain, Xtest, features: str, ngram: tuple):
    print("\nVectorizing data...")
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                            analyzer=features, ngram_range=ngram, max_features=3000)
    scaler = StandardScaler(with_mean=False)

    # Training data
    Xtrain_tfidf = tfidf.fit_transform(Xtrain)
    X_train = scaler.fit_transform(Xtrain_tfidf)

    # Test data
    Xtest_tfidf = tfidf.transform(Xtest)
    X_test = scaler.transform(Xtest_tfidf)

    return X_train, X_test


def search_grid(X_train):
    # variate your nu parameter to decrease number of outliers
    # nu: margin, corresponds to the probability of finding a new, but regular, observation outside the frontier.
    ocsvm_parameters = {'nu': [0.05, 0.01, 0.1], 'gamma': ['auto', 'scale']}

    # assuming -1 is the label for the target class
    scoring_function = make_scorer(f1_score, pos_label=1)

    print("\nGrid searching for OCSVM... ")
    grid_model = GridSearchCV(
        ocsvm, ocsvm_parameters, n_jobs=-1, cv=5, verbose=2, scoring=scoring_function)
    grid_model.fit(X_train, y_train)
    print("Best params: ", grid_model.best_params_,
          "\nwith Score: ", grid_model.best_score_)


def run_model(X_train, X_test):
    print("\nTraining OCSVM model...")
    ocsvm.fit(X_train)
    y_pred = ocsvm.predict(X_test)
    # y_pred_train = ocsvm.predict(X_train)
    # y_pred_outliers = ocsvm.predict(X_outliers)
    # n_error_train = y_pred_train[y_pred_train == -1].size
    # n_error_test = y_pred[y_pred == -1].size
    # n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    print(confusion_matrix(y_test, y_pred, normalize='true'))
    return y_pred


# Load Data
title = 'Parsed_LLw46_OneClassSVMnu0.005'

file = "data/benchmarkData/PARSEDdata.csv"
df = pd.read_csv(file, sep=';')
fother = "data/benchmarkData/PARSEDDataotherAuthors.csv"
df2 = pd.read_csv(fother)
# df = pd.concat([df1, df2], ignore_index=True, axis=0)
X_train, y_train, X_test, y_test, test_authors = load_data(
    df, labels='Law|Late')  # labels: what to include in positive_class
# The training data is not polluted by outliers!

# Vectorize Data
features = 'word'  # 'word' for PARSED
ngram = (4, 6)  # (3,6)
X_train, X_test = vectorize_data(
    X_train, X_test, features, ngram)

# Load Model
# Unsupervised Outlier Detection.
ocsvm = OneClassSVM(kernel="rbf", gamma='scale', nu=0.005)

# GridSearch
# search_grid(X_train)
# exit(0)

# Fit and Predict with best parameters
y_pred = run_model(X_train, X_test)

for idx in range(len((y_pred))):
    print(f"Predicted {y_pred[idx]} for Author: {test_authors[idx]}")


# Plot Model
plot_unsupervised_model(X_train, X_test, title, xlim=(
    0, 200), ylim=(-50, 50), document_names=test_authors)
