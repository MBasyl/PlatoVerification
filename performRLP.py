import numpy as np
import sys
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from models.RLP import *
from collections import Counter
import matplotlib.pyplot as plt


def grid_search(df, data):
    # Perform multiple iterations to find best hyperparams
    grams = [4, 5, 6, 8]
    length = [500, 1000, 3000]
    for n in grams:
        for L in length:
            model = RLP(n=n, L=L, char=False)
            # note down parameters of experiment
            print("PARAMS : ", {data}, {n}, {L})
            train_test(model, df, data)
            print("\n\n")


def create_dataframe(y_test, y_probas):
    data = []
    for true_label, (predicted, distance1, gap) in zip(y_test, y_probas):
        data.append({
            'true_label': true_label,
            'predicted': predicted,
            'distance1': distance1,
            'gap': gap
        })
    return pd.DataFrame(data)


def cross_validate(documents, classes, model, n=5):
    # First, setup the cross fold validation object
    cv = KFold(n_splits=n, random_state=None)
    scores = cross_val_score(model, documents, classes, cv=cv)
    print("Mean Accuracy over 5 folds is {:.1f}%".format(
        100. * np.mean(scores)))


def train_test(model, df: pd.DataFrame, data: str):

    # Train-test split, keeping class imbalance
    X = df['text']
    y = df['label']
    doc_titles = df['title'].to_list()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3, stratify=y)
    # Get original indexes for later insight
    test_idx = X_test.index

    print("Class balances", Counter(y_train), Counter(y_test))

    # cross-val first
    cross_validate(X.to_list(), np.array(y), model, n=5)
    # Training the model
    model.fit(X_train.to_list(), np.array(y_train))
    y_pred = model.predict(X_test.to_list())
    y_probas = model.predict_distance(X_test.to_list())

    df_distance = create_dataframe(y_test, y_probas)
    df_distance.to_csv(f"{data}distance_predictions.csv", index=False)
    # Evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(
        y_test, y_pred, normalize='true'))

    print("Classification Report:\n", classification_report(y_test, y_pred))
    # Print ground truth and predicted authors for misclassified documents
    # df = df[~df['title'].str.contains('VII|#')]

    for idx, y_pred_val, y_test_val in zip(test_idx, y_pred, y_test):
        try:
            if y_pred_val != y_test_val:
                print(
                    f"Misclassified Title: {df['title'].iloc[idx]}, at index: {idx}")
            else:
                print(
                    f"Correct {df['title'].iloc[idx]} with label {y_pred_val}")
        except IndexError:
            print(f"\n\nIndex out of bounds for index: {idx}\n")

    return y_pred, y_probas, doc_titles


def plot_features(feature_df, title, filename, ascending=True, top_n=15):
    # Sort the DataFrame by coefficient magnitude
    feature_df = feature_df.reindex(
        feature_df['Coefficient'].abs().sort_values(ascending=ascending).index)

    # Display the top N features
    top_features = feature_df.head(top_n)

    # Create a bar plot
    plt.figure(figsize=(15, 12))
    plt.barh(top_features['Feature'],
             top_features['Coefficient'], color='skyblue')
    plt.xlabel('Coefficient')
    plt.title(title, weight='bold')
    plt.savefig(filename, bbox_inches='tight')


def feature_importance(rlp_model, savepath):
    feature_importance = {}

    support_vectors = rlp_model.author_profiles
    coefficients = {author: rlp_model.compare_profiles(
        rlp_model.language_profile, profile) for author, profile in support_vectors.items()}

    # Iterate through each support vector
    for label, vector in support_vectors.items():
        importance = 0.0
        # Multiply each feature's value by its corresponding coefficient and accumulate
        for feature, value in vector.items():
            importance += value * coefficients[label]
            feature_importance[feature] = importance
    feature_df = pd.DataFrame(
        {'Coefficient': feature_importance.values(), 'Feature': feature_importance.keys()})

    # Plot least representative features
    plot_features(feature_df, 'Least Representative Features',
                  f'{savepath}bottmom.png', ascending=True)

    # Plot top representative features
    plot_features(feature_df, 'Top Representative Features',
                  f'{savepath}top.png', ascending=False)


def plot_author_features(df, title, filename, top_n=15):
    # Display the top N features
    top_features = df.groupby('Author').apply(
        lambda group: group.nlargest(top_n, 'Coefficient')).reset_index(level='Author', drop=True)
    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=1, ncols=2,
                             figsize=(12, 10))
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    # Iterate through each author and plot the corresponding data
    for i, (author, group) in enumerate(top_features.groupby('Author')):
        axes[i].barh(group['Feature'], group['Coefficient'], color='skyblue')
        axes[i].set_xlabel('Coefficient Values')
        axes[i].set_title(author)
    fig.suptitle(title, weight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{filename}.png', bbox_inches='tight')


def profile_feature_importance(rlp_model, savepath):
    feature_importance = {}
    support_vectors = rlp_model.author_profiles
    coefficients = {author: rlp_model.compare_profiles(
        rlp_model.language_profile, profile) for author, profile in support_vectors.items()}
    # Iterate through each support vector
    for label, vector in support_vectors.items():
        # Multiply each feature's value by its corresponding coefficient and accumulate
        importance = sum(value * coefficients[label]
                         for value in vector.values())
        # Store feature importance per author
        author_feature_importance = {
            feature: importance * value for feature, value in vector.items()}
        feature_importance.update(author_feature_importance)

    feature_df = pd.DataFrame({'Author': label, 'Coefficient': feature_importance.values(
    ), 'Feature': feature_importance.keys()})
    # Sort the DataFrame by coefficient
    feature_df = feature_df.reindex(
        feature_df['Coefficient'].abs().sort_values(ascending=False).index)

    inverted_author_mapping = {0: 'Not-Plato', 1: 'Plato'}
    # inverted_author_mapping = {
    #     0: 'Xen_Histories',
    #     1: 'AlcibiadesII',
    #     2: 'Hipparcus',
    #     3: 'AlcibiadesI',
    #     4: 'Xen_Dialogues',
    #     5: 'Theages',
    #     7: 'Pla_Laws',
    #     8: 'Lovers',
    #     9: 'Pla_Late',
    #     10: 'Minos',
    #     11: 'Pla_Ealy',
    #     12: 'Pla_Mature',
    #     13: 'HippiasMajor'
    # }
    feature_df['Author'] = feature_df['Author'].map(
        inverted_author_mapping.get)
    df = feature_df[~feature_df['Feature'].str.contains('\*|(PROPN)')]
    df.to_csv(f"{savepath}df_featimport.csv", index=False)

    plot_author_features(df, 'Top Representative Features',
                         filename=savepath, top_n=15)


def visualize_author_distances(testdata, y_pred, distance_pred, classes):
    """NOT used, works only with MULTI-LABELS not binary?"""
    testdata['title'] = testdata['title'].str.replace('Law\d+', 'Law')
    classes = list(testdata['author'])
    # Visualize distances between documents and predicted authors in a scatter plot
    plt.figure(figsize=(10, 6))

    for idx, label in enumerate(y_pred):
        distances = [float(dist) for author, dist in distance_pred[idx]]

        # Scatter plot for each document
        print("classes", [classes[idx]], "\nLEN distances: ", len(distances))
        print(distances)
        plt.scatter([classes[idx]] * len(distances), distances,
                    label=label)

    plt.title('Distances between Documents and Predicted Authors')
    plt.xlabel('Authors')
    plt.ylabel('Distance')
    plt.legend()
    # plt.show()


if __name__ == "__main__":

    sys.stdout = open(f'RLP_log.txt', 'a')

    #################################
    # dataset path
    data = 'PARSED'
    df = pd.read_csv(f"data/{data}dataset_obfuscate.csv")
    # Validation set
    val_set = df[df['title'].str.contains('VII|#')]
    df = df[~df['title'].str.contains('VII|#')]

    # GridSearch for parameter-tuning
    # grid_search(df, data)

    # Best param PLAIN with LATE AND LAWS
    # {6, 1000}  F1 0.73 (SVM)
    # {8, 1000}  F1 0.76 --> 9-1000 F1 0.83 --> WORD TRIGRAMS: 0.87 **

    # Best param PARSED with LATE AND LAWS
    # {4, 1000}  F1 0.67 (SVM)
    # {5, 1000}  F1 0.76
    model = RLP(n=5, L=1000, char=False)
    y_pred, y_probas, doc_titles = train_test(model, df, data)

    # Visualize feature importance
    feature_importance(model, savepath=f'{data}featureImportance')
    profile_feature_importance(
        model, savepath=f'{data}AuthorfeatureImportance')
