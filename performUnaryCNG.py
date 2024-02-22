import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from models.UnaryCNG import UnaryCNG as cng
from collections import Counter
import matplotlib.pyplot as plt


def create_dataframe(y_test, y_pred, distances, threshold):

    df = pd.DataFrame(distances, columns=['cosine', 'correlation'])
    df['true_label'] = list(y_test)
    df['prediction'] = list(y_pred)

    # Calculate the differences
    differences = distances - threshold[1]
    df['difference_cosine'] = differences[:, 0]
    df['difference_correlation'] = differences[:, 1]
    return df


def cross_validate(model, df, save_path):
    # print("###########\nusing: ", save_path, model)
    X = df['text']
    y = df['label']
    cv = StratifiedKFold(n_splits=5)
    accuracy_scores = []
    f1_scores = []

    for train_index, val_index in cv.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Fit the estimator on the training data
        model.fit(X_train, y_train)

        # Predict on the validation data
        y_pred = model.predict(X_val)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
    print(np.mean(accuracy_scores), np.mean(f1_scores))

    plt.figure(figsize=(12, 9))
    folds = range(1, len(accuracy_scores) + 1)
    plt.plot(folds, accuracy_scores, marker='o', label='Accuracy')
    plt.plot(folds, f1_scores, marker='o', label='F1 Score')
    plt.title('Accuracy and F1 Score Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}.png")


def grid_search():
    data = ['PARSED', 'PLAIN']
    ngram = [3, 4, 5, 6, 8]
    length = [500, 1000, 3000]
    measure = ['cosine', 'minmax']
    for d in data:
        df = pd.read_csv(f"data/svmDataset/{d}dataset_obfuscate.csv")
        df = df[~df['title'].str.contains('VII|#')]
        if d == 'PARSED':
            char = False
        else:
            char = True
        for m in measure:
            for n in ngram:
                for L in length:
                    print("using: ", n, L, char, m, d)
                    model = cng(n=n, L=L, char=char)
                    train_test(
                        model, df, m, data)


def train_test(model, df: pd.DataFrame, measure='cosine'):

    # Train-test split, keeping class imbalance
    testdata = df[df['title'].str.contains('test')]
    X_test = testdata['text']
    y_test = testdata['label']

    traindata = df[~df['title'].str.contains('test')]
    X_train = traindata['text']
    y_train = traindata['label']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)  # seed3

    print("Class balances", Counter(y_train), Counter(y_test))

    # Training the model
    fitted_threshold = model.fit(documents=X_train, classes=y_train)
    y_pred, distances = model.predict(X_test)

    # Evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(
        y_test, y_pred, normalize='true'))
    print(f"Classification Report:\n",
          classification_report(y_test, y_pred))

    # Print ground truth and predicted authors for misclassified documents
    doc_titles = testdata['title'].to_list()
    y_test.reset_index(drop=True, inplace=True)
    for idx, label in enumerate(y_pred):
        status = "Misclassified" if label != y_test[idx] else "Correct"
        print(
            f"{status} - Predicted {label} for document: {doc_titles[idx]}")

    # Get fine-grained values
    df_distance = create_dataframe(y_test, y_pred, distances, fitted_threshold)
    df_distance['documents'] = doc_titles
    df_distance.to_csv(f"{data}distance_predictions.csv", index=False)


if __name__ == "__main__":

    # sys.stdout = open(f'final_unarRLP_log.txt', 'a')

    # dataset path
    data = 'PLAIN'
    df = pd.read_csv(f"{data}personal_obfuscate.csv", sep=";")
    #################################

    # Remove validation set
    df = df[~df['title'].str.contains('VII|val')]
    df = df.reset_index(drop=True)
    ################################
    # Choose best parameters
    # grid_search()
    #################################
    chosen_params = {'n': 8, 'L': 500, 'char': True}
    model = cng(**chosen_params)
    # Perform cross-validation
    # cross_validate(model, df, save_path=f"{data}cvScoresLaws{n}")

    train_test(model, df, data)

    # Visualize feature importance
    model.plot_feature_importance(
        title='Top Features for PLAIN model', filename=f'{data}featImportance.png')


# CV RESULTS (cosine):
# using:  PLAIN 8 500 onlyLAWS: 0.97 0.82
# using:  PARSED 3 3000 onlyLAWS: 0.94 0.64
# using:  PARSED 3 3000: 0.78 0.71 *
# using:  PARSED 4 3000: 0.79 0.71
# using:  PLAIN 8 500: 0.86 0.69 *
