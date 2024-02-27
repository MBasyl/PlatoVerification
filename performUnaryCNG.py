import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from models.UnaryCNG import UnaryCNG as cng
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def scatterplot(df):
    """
    Statistical Tests: Conduct statistical tests to determine if the observed differences in mean distances between 
    correctly and incorrectly classified instances are statistically significant. You can use t-tests, ANOVA, or 
    non-parametric tests depending on the distribution of your data and the assumptions you can make.
    """
    # Prettify labels for plot
    df['documents'] = df['documents'].str.replace('#test', '')
    # Set style
    sns.set_theme(style="whitegrid")
    fig = plt.subplots(figsize=(15, 15))
    sns.pairplot(df, vars=['cosine', 'correlation'],
                 hue='true_label', diag_kind='kde', kind='scatter')

    plt.suptitle(
        f'Pairplot of Distances and True Labels for {data} data', y=1, fontweight='bold')
    plt.savefig(f"{data}trainPairDistance.png", bbox_inches='tight')

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(data=df, x='cosine', y='correlation',
                    hue='true_label', s=100, ax=ax)  # style='prediction',
    # Add document names next to data points
    for i in range(len(df)):
        ax.text(df['cosine'][i], df['correlation']
                [i], df['documents'][i], fontsize=8)

    # Add title and labels
    plt.title('Cosine vs. Cole')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Cole Correlation')
    plt.legend(title='True Label')
    plt.tight_layout()
    plt.savefig(f"{data}trainCNGscatter.png")


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
    print("Fitted Threshold", fitted_threshold)
    # y_pred, distances = model.predict(X_test)
    y_pred, distances = model.predict(X_train)

    # Evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(
        y_train, y_pred, normalize='true'))
    print(f"Classification Report:\n",
          classification_report(y_train, y_pred))

    # Print ground truth and predicted authors for misclassified documents
    doc_titles = traindata['title'].to_list()
    y_train.reset_index(drop=True, inplace=True)
    for idx, label in enumerate(y_pred):
        status = "Misclassified" if label != y_train[idx] else "Correct"
        print(
            f"{status} - Predicted {label} for document: {doc_titles[idx]}")

    # Get fine-grained values
    df_distance = create_dataframe(
        y_train, y_pred, distances, fitted_threshold)
    df_distance['documents'] = doc_titles
    # Visualize results and save figure
    scatterplot(df_distance)
    # Save df
    df_distance.to_csv(f"{data}distance_predictions2.csv", index=False)


if __name__ == "__main__":

    sys.stdout = open(f'unaryCNG_log.txt', 'a')

    # dataset path
    data = 'PARSED'
    df = pd.read_csv(f"{data}personal_obfuscate.csv", sep=";")
    #################################

    # Remove validation set
    df = df[~df['title'].str.contains('VII|val')]
    df = df.reset_index(drop=True)
    ################################
    # Choose best parameters
    # grid_search()
    #################################
    chosen_params = {'n': 4, 'L': 3000, 'char': False}
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
# using:  PARSED 4 3000: 0.79 0.71 * better on full dataset compared to 3
# using:  PLAIN 8 500: 0.86 0.69 *
