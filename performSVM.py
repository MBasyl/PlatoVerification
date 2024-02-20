import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import RocCurveDisplay, auc
import matplotlib.pyplot as plt
import seaborn as sns
import sys


class SVMModel:
    def __init__(self, df: pd.DataFrame):
        sys.stdout = open(f'outputs/SVM/SVM_log.txt', 'a')
        self.df = df

    def process_data(self, drop_history=False):
        # print("Balance: ", df.label.value_counts()) # output: 135/33
        if drop_history:
            # Drop 'Histories' rows, balance 87/33
            self.df.drop(
                self.df[self.df.title == 'Histories'].index, inplace=True)

        # Validation set
        val_set = self.df[self.df['title'].str.contains('VII|#')]
        X_val = val_set['text']
        y_val = val_set['label']

        # Train-test split, keeping class imbalance
        self.df = self.df[~self.df['title'].str.contains('VII|#')]

        X = self.df['text']
        y = self.df['label']
        doc_names = self.df['title']
        return X_val, y_val, X, y, doc_names

    def perform_SearchGrid(self, X, y, feature: str, save_path: str):
        """
        feature: ['word', 'char'].
        save_path: filename to save plots and output df.
        Code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html
        """

        print("\nPerforming GridSearch, might take a while...\n")
        # use two underscores between estimator name and its parameters
        param_grid = {
            "vect__max_df": [1, 0.4, 0.8],
            "vect__min_df": [1, 5],
            "vect__ngram_range": [(4, 5), (3, 6), (4, 8)],
            "vect__max_features": [1000, 3000, 5000],
            "clf__kernel": ["linear", "rbf"],
            # Not influent:
            # --> balanced: [0.72 1.62]
            "clf__class_weight": [None, 'balanced']
        }
        pipeline = Pipeline(
            [("vect", TfidfVectorizer(strip_accents=None, lowercase=False,
                                      analyzer=feature)),  # , max_features=1000
             ("clf", SVC(C=100, random_state=0))
             ])
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=0)
        search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                              scoring="f1", cv=cv, verbose=2, n_jobs=-1)
        search.fit(X, y)
        results_df = pd.DataFrame(search.cv_results_)
        results_df = results_df.sort_values(by=["rank_test_score"])
        results_df = results_df.set_index(
            results_df["params"].apply(lambda x: "_".join(str(val)
                                                          for val in x.values()))
        ).rename_axis("kernel")

        savedf = results_df[["params", "rank_test_score",
                            "mean_test_score", "std_test_score", "mean_fit_time"]]
        savedf.to_csv(f"{save_path}.csv", index=False)

        # create df of model scores ordered by performance
        model_scores = results_df.filter(regex=r"split\d*_test_score")

        # plot 30 examples of dependency between cv fold and AUC scores
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.lineplot(
            data=model_scores.transpose().iloc[:30],
            dashes=False,
            palette="Set1",
            marker="o",
            alpha=0.5,
            ax=ax,
        )
        ax.set_xlabel("CV test fold", size=12, labelpad=10)
        ax.set_ylabel("Model F1 score", size=12)
        ax.tick_params(bottom=True, labelbottom=False)
        # Move the legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def perform_crossval(self, X, y, feature: str, save_path: str):

        tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                                analyzer=feature,
                                ngram_range=(4, 4),
                                max_features=1000, max_df=0.4)
        scaler = StandardScaler(with_mean=False)

        X_train_tfidf = tfidf.fit_transform(X)
        X = scaler.fit_transform(X_train_tfidf)
        # n_samples, n_features = X.shape

        model = SVC(kernel='rbf', C=100, probability=True, random_state=54)

        # Train the model
        classifier = model.fit(X, y)
        cv = StratifiedKFold(n_splits=5)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(6, 6))
        for fold, (train, test) in enumerate(cv.split(X, y)):
            # Convert sparse matrices to arrays
            X_train_fold = X[train].toarray()
            y_train_fold = y.iloc[train]  # Access labels using iloc
            X_test_fold = X[test].toarray()
            y_test_fold = y.iloc[test]  # Access labels using iloc
            print("\nTRAIN/TEST balance: ", Counter(y_train_fold),
                  "\t", Counter(y_test_fold))

            # Fit the model on training fold
            classifier.fit(X_train_fold, y_train_fold)

            # classifier.fit(X[train], y[train])
            viz = RocCurveDisplay.from_estimator(
                classifier,
                X_train_fold,
                y_train_fold,
                name=f"ROC fold {fold}",
                alpha=0.3,
                lw=1,
                ax=ax,
                plot_chance_level=True,  # (fold == 5 - 1)
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability\n(Positive label 'Plato')",
        )
        ax.legend(loc="lower right")
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        plt.show()

        pass

    def execute_model(self, X, y, feature, ngrams):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=3, stratify=y)

        model = make_pipeline(
            TfidfVectorizer(strip_accents=None, lowercase=False,
                            analyzer=feature,  # "vect__min_df": [1],
                            ngram_range=ngrams,
                            max_features=1000),
            StandardScaler(with_mean=False),
            SVC(kernel='rbf', C=100)
        )

        # Train the model
        cls = model.fit(X_train, y_train)
        return cls, X_test, y_test

    def get_predictions(self, cls, X_test, y_test):
        # Make y_pred on the test set
        y_pred = cls.predict(X_test)
        probabilities = cls.decision_function(X_test)

        # Evaluate the model
        print("Confusion Matrix:\n", confusion_matrix(
            y_test, y_pred, normalize='true'))

        print("Classification Report:\n", classification_report(y_test, y_pred))
        # Print ground truth and predicted authors for misclassified documents
        df = self.df[~self.df['title'].str.contains('VII|#')]
        misclass = np.where(y_pred != y_test)[0]
        for i, idx in enumerate(misclass):
            print(
                f"Misclassified Title: {df['title'].iloc[idx]}, at index: {idx}")

    @staticmethod
    def reduce_dimentionality(X_train, y_train, feature, ngrams):
        tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                                analyzer=feature,
                                ngram_range=ngrams,
                                max_features=3000)
        scaler = StandardScaler(with_mean=False)
        svd = TruncatedSVD(n_components=2)
        model = SVC(kernel='linear', C=100)
        # Train data
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_train = scaler.fit_transform(X_train_tfidf)
        X_svd = svd.fit_transform(X_train)
        feature_names = tfidf.get_feature_names_out()

        # Test data
        # X_test_tfidf = tfidf.transform(X_test)
        # X_test = scaler.transform(X_test_tfidf)
        # X_svd_test = svd.transform(X_test)

        cls = model.fit(X_svd, y_train)

        return cls, feature_names, X_svd, y_train

    def get_feature_importance(self, X, y, params):

        model, feature_names, _, _ = self.reduce_dimentionality(
            X, y, **params)
        # Extract dual coefficients directly from the SVM model
        dual_coefficients = model.dual_coef_.flatten()

        # Create a DataFrame with feature names and coefficients
        feature_df = pd.DataFrame(
            {'Feature': feature_names, 'Coefficient': 0.0})

        # Update coefficients for features in support vectors
        for i, sv_index in enumerate(model.support_):
            feature_df.loc[sv_index, 'Coefficient'] += dual_coefficients[i]

        # Sort the DataFrame by coefficient magnitude
        feature_df = feature_df.reindex(
            feature_df['Coefficient'].abs().sort_values(ascending=False).index)

        # Plot the top N features and their coefficients
        top_n = 50  # Adjust as needed
        top_features = feature_df.head(top_n)

        plt.figure(figsize=(15, 10))
        ax = plt.subplot(111)  # Create subplot
        ax.barh(top_features['Feature'],
                top_features['Coefficient'], color='skyblue')

        # Set margin on the left side
        plt.subplots_adjust(left=0.1)

        # Inscribing labels inside bars
        for index, (feature, coefficient) in enumerate(zip(top_features['Feature'], top_features['Coefficient'])):
            plt.text(coefficient, index, feature, va='center')

        plt.xlabel('Coefficient')
        plt.title(f'Top {top_n} Features')
        plt.show()

    def viz_decision_boundaries(self, X, y, save_path, document_names=None):

        model, _, X_train, y_train = self.reduce_dimentionality(
            X, y, **params)

        # Plot the data points
        plt.scatter(X_train[:, 0], X_train[:, 1], c=list(y_train),
                    cmap='viridis', edgecolor='k', s=40)
        if document_names:
            # Label scattered dots with document names
            for i, name in enumerate(document_names):
                plt.text(X_train[i, 0], X_train[i, 1],
                         name, fontsize=8, ha='right')

        plt.title('Decision Boundary Projection')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        # Plot decision boundary
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                             np.linspace(ylim[0], ylim[1], 50))
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.contour(xx, yy, Z, colors='k', levels=[
                    0], alpha=0.5, linestyles=['-'])
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    data = 'PLAIN'
    feature = 'char'
    csv_data = f"data/svmDataset/{data}dataset_obfuscate.csv"

    df = pd.read_csv(csv_data)
    svm_model = SVMModel(df)
    X_val, y_val, X, y, doc_names = svm_model.process_data(
        drop_history=True)

    # [94, 26]/[24, 6]
    params = {
        'Type of data': data,
        'Dataset balance': Counter(y).values(),
        'validation': len(list(y_val)),
        'exclude histories': True,
        'include Late': True,
        'feature': feature,
        'ngrams': (3, 8),
    }

    # Gridsearch for best hyperparameters
    svm_model.perform_SearchGrid(
        X, y, feature=feature, save_path=f'{data}gridSVM')
    # PARSED: {'clf__kernel': 'linear', 'vect__max_df': 0.4, 'vect__ngram_range': (4, 4)}
    # PLAIN:
    exit(0)
    # Apply best hyperparams and get mean results from Cross-Validation
    # svm_model.perform_crossval(X, y, feature = feature, savepath = f'{data}SVMcvROC')

    # Train and test model
    model, X_test, y_test = svm_model.execute_model(X, y, **params)
    svm_model.get_predictions(model, X_test, y_test)

    # Inspect features. i.e. start of sentences (use periods) influent?

    # Inspect features: not very informative
    # SHAP??
    svm_model.get_feature_importance(X, y, **params)

    # Visualize decision boundaries
    svm_model. viz_decision_boundaries(X, y,
                                       save_path='svmplot2.png', document_names=doc_names)
