import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from models.UnaryCNG import UnaryCNG as cng
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def plot_roc_curve(y_test, y_probs, save_path):
    print("AUC score: ", roc_auc_score(y_test, y_probs))

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"{save_path}.png")
    plt.show()


def plotting(df):
    """
    Statistical Tests: Conduct statistical tests to determine if the observed differences in mean distances between 
    correctly and incorrectly classified instances are statistically significant. You can use t-tests, ANOVA, or 
    non-parametric tests depending on the distribution of your data and the assumptions you can make.
    """
    # Prettify labels for plot
    df['documents'] = df['documents'].str.replace('#test', '')
    # Set style
    sns.set_theme(style="whitegrid")

    # VIOLIN PLOT
    plt.figure(figsize=(15, 12))
    ax = sns.violinplot(x='true_label', y='cosine', data=df,
                        inner='quartile', palette='pastel')

    # Increase font size for labels and ticks
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticklabels(["Not-Plato", "Plato"], fontsize=26, weight='bold')

    # Label quartiles
    # Add quartile labels
    for i in range(2):
        quartile_vals = np.percentile(
            df[df['true_label'] == i]['cosine'], [25, 50, 75])
        ax.text(i, quartile_vals[0], f'Q1: {quartile_vals[0]:.2f}',
                ha='center', va='bottom', color='black', fontsize=14)
        ax.text(i, quartile_vals[1], f'Q2: {quartile_vals[1]:.2f}',
                ha='center', va='bottom', color='black', fontsize=14)
        ax.text(i, quartile_vals[2], f'Q3: {quartile_vals[2]:.2f}',
                ha='center', va='bottom', color='black', fontsize=14)

    # Add labels and adjust fontsize
    ax.set_title('Cosine Distance Distribution', fontsize=30, weight='bold')
    ax.set_xlabel('True Label', fontsize=14)
    ax.set_ylabel('Cosine Distance', fontsize=14)
    plt.tight_layout()
    plt.savefig("cosine_violinplot.png")
    plt.close()

    # SCATTER PLOT:
    plt.figure(figsize=(15, 12))
    ax = sns.scatterplot(data=df, x=np.abs(df['cosine']), y=np.abs(df['cosine']),
                         hue='true_label', style='true_label', s=200)

    # Increase font size for labels
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Add document names next to data points'
    for i in range(len(df)):
        if df['documents'][i] in ['Epi', 'Tim', 'Par', 'Histories', 'Dialogues']:
            # offset_x = 0.02  # adjust this value to set the horizontal offset
            # offset_y = 0.02  # adjust this value to set the vertical offset
            ax.text(np.abs(df['cosine'][i]), np.abs(df['cosine']
                    [i]), df['documents'][i], fontsize=16)

    # Add title and adjust fontsize
    plt.title('Cosine Distribution', fontsize=30, weight='bold')

    # Add labels
    plt.xlabel('Cosine Distance from Plato profile', fontsize=14)
    plt.ylabel('Cosine Difference from Threshold', fontsize=14)

    plt.legend(title='True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig("cosine_scatterplot.png")
    plt.close()


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
    print("###########\nusing: ", save_path, model)
    X, y = df['text'], df['label']
    cv = StratifiedKFold(n_splits=5)
    accuracy_scores = []
    f1_scores = []

    for train_index, val_index in cv.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Fit & predict
        model.fit(X_train, y_train)
        y_pred, _ = model.predict(X_val, measure='cosine')

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
    data = ['PLAIN']  # 'PARSED',
    ngram = [3, 4, 5, 6, 8]
    length = [500, 1000, 3000]
    measure = ['cosine', 'cole']
    for d in data:
        df = pd.read_csv(f"{d}personal_obfuscate.csv", sep=";")
        df = df[~df['title'].str.contains('VII|val')]
        df = df.reset_index(drop=True)
        X, y, titles = df['text'], df['label'], df['title']
        if d == 'PARSED':
            char = False
        else:
            char = True
        for m in measure:
            for n in ngram:
                for L in length:
                    print("using: ", n, L, char, m, d)
                    model = cng(n=n, L=L, char=char)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=3, stratify=y)

                    print("Class balances", Counter(y_train), Counter(y_test))

                    # Training the model
                    model.fit(X_train, y_train)
                    y_pred, _ = model.predict(X_test, m)

                    # Evaluate the model
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    print("Accuracy: ", accuracy, "F1: ", f1)
                    print()


def train_test(model, df, data, measure='cosine'):
    # Train-test split, keeping class imbalance
    testdata = df[df['title'].str.contains('test')]
    X_test, y_test = testdata['text'], testdata['label']
    doc_titles = testdata['title'].to_list()

    traindata = df[~df['title'].str.contains('test')]
    X_train, y_train = traindata['text'], traindata['label']

    print("Class balances", Counter(y_train), Counter(y_test))

    # Training the model
    fitted_threshold = model.fit(documents=X_train, classes=y_train)
    y_pred, distances = model.predict(X_test, measure)

    # Evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(
        y_test, y_pred, normalize='true'))
    print(f"Classification Report:\n",
          classification_report(y_test, y_pred))

    # Print ground truth and predicted authors for misclassified documents
    y_test.reset_index(drop=True, inplace=True)
    for idx, label in enumerate(y_pred):
        status = "Misclassified" if label != y_test[idx] else "Correct"
        print(
            f"{status} - Predicted {label} for document: {doc_titles[idx]}")

    # Get fine-grained values
    df_distance = create_dataframe(
        y_test, y_pred, distances, fitted_threshold)
    df_distance['documents'] = doc_titles
    df_distance.to_csv(f"{data}distance_predictions.csv", index=False)

    # Visualize results and save figure
    plotting(df_distance)

    # Visualize feature importance
    model.plot_feature_importance(
        title='Top Features for PARSED model', filename=f'CNG{data}Features.png')


def perform_validation(model, df, val_set, data):
    X_train, y_train = df['text'], df['label']
    X_test, y_test, val_titles = val_set['text'], val_set['label'], val_set['title'].to_list(
    )
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
    y_test.reset_index(drop=True, inplace=True)
    for idx, label in enumerate(y_pred):
        status = "Misclassified" if label != y_test[idx] else "Correct"
        print(
            f"{status} - Predicted {label} for document: {val_titles[idx]}")

    # Get fine-grained values
    df_distance = create_dataframe(
        y_test, y_pred, distances, fitted_threshold)
    df_distance['documents'] = val_titles
    df_distance.to_csv(f"{data}Val_distance_predictions.csv", index=False)

    # Visualize results and save figure
    plotting(df_distance)

    # Visualize feature importance
    model.plot_feature_importance(
        title='Top Features for PARSED model', filename=f'CNG{data}ValFeatures.png')


if __name__ == "__main__":

    # sys.stdout = open(f'CNG_log.txt', 'a')
    ################################
    # Choose best parameters
    # grid_search()

    ################################
    # dataset path
    data = 'PARSED'
    df = pd.read_csv(f"{data}personal_obfuscate.csv", sep=";")
    #################################
    # Remove validation set
    df = df[~df['title'].str.contains('VII|val')]
    df = df.reset_index(drop=True)

    chosen_params = {'n': 3,
                     'L': 3000,
                     'char': False}
    model = cng(**chosen_params)
    # Perform cross-validation
    # cross_validate(model, df, save_path=f"{data}cvScores")
    # exit(0)
    #################################

    # train_test(model, df, data)
    # exit(0)
    #################################
    csv_data = f"{data}dataset_validation.csv"
    val_set = pd.read_csv(csv_data, sep=";")
    perform_validation(model, df, val_set, data)
