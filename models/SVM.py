import pandas as pd
import numpy as np
import sys
import shap
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class SVMModel:
    def __init__(self, df: pd.DataFrame):
        sys.stdout = open(f'SVM_log.txt', 'a')
        self.df = df

    def process_data(self, drop_history=False):
        # print("Balance: ", df.label.value_counts()) # output: 135/33
        if drop_history:
            # Drop 'Histories' rows, balance 87/33
            self.df.drop(
                self.df[self.df.title == 'Histories'].index, inplace=True)

        # Validation set
        val_set = self.df[self.df['title'].str.contains('VII|\.')]
        X_val, y_val = val_set['text'], val_set['label']

        # Train-test split, keeping class imbalance
        self.df = self.df[~self.df['title'].str.contains('VII|\.')]
        X, y = self.df['text'], self.df['label']

        return X_val, y_val, X, y

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
            # "vect__min_df": [1, 5],
            "vect__ngram_range": [(4, 4), (4, 5), (5, 6), (3, 6)],
            # "vect__max_features": [1000, 3000],
            "clf__kernel": ["linear"]  # , "rbf"],
            # Not influent: "clf__class_weight": [None, 'balanced']
        }
        pipeline = Pipeline(
            [("vect", TfidfVectorizer(strip_accents=None, lowercase=False, max_features=1000,
                                      analyzer=feature, token_pattern=r"(?u)\b\w+(?:[-:]\w+)?(?:[-:]\w+)?\b")),
             ("clf", SVC(C=100, random_state=0))
             ])
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
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

    def perform_crossval(self, X, y, save_path: str, feature, ngrams, max_features, max_cull):

        tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                                analyzer=feature,
                                ngram_range=ngrams, token_pattern=r"(?u)\b\w+(?:[-:]\w+)?(?:[-:]\w+)?\b",
                                max_features=max_features, max_df=max_cull)
        scaler = StandardScaler(with_mean=False)

        X_train_tfidf = tfidf.fit_transform(X)

        X = scaler.fit_transform(X_train_tfidf)
        # n_samples, n_features = X.shape

        model = SVC(kernel='linear', C=100, probability=True, random_state=54)

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

    def execute_model(self, feature, ngrams, max_features, max_cull):
        # Train-test split, keeping class imbalance
        traindata = self.df[~self.df['title'].str.contains('test')]
        X_train, y_train = traindata['text'], traindata['label']
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=3, stratify=y)
        pipe = Pipeline(
            [("vect", TfidfVectorizer(strip_accents=None, lowercase=False,
                                      analyzer=feature,
                                      ngram_range=ngrams,
                                      max_features=max_features,
                                      max_df=max_cull,
                                      token_pattern=r"(?u)\b\w+(?:[-:]\w+)?(?:[-:]\w+)?\b")),
             ("scalar", StandardScaler(with_mean=False)),
             ("clf", SVC(C=100, kernel='linear'))
             ])

        # Train the model
        model = pipe.fit(X_train, y_train)
        # Getting feature names
        feature_names = pipe.named_steps['vect'].get_feature_names_out()

        return model, X_train, y_train, feature_names

    def get_predictions(self, model):
        testdata = self.df[self.df['title'].str.contains('test')]
        X_test, y_test = testdata['text'], testdata['label']

        # Make y_pred on the test set
        y_pred = model.predict(X_test)
        probabilities = model.decision_function(X_test)

        # Evaluate the model
        print("Confusion Matrix:\n", confusion_matrix(
            y_test, y_pred, normalize='true'))

        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Print ground truth and predicted authors for misclassified documents
        doc_titles = testdata['title'].to_list()
        y_test.reset_index(drop=True, inplace=True)
        for idx, label in enumerate(y_pred):
            status = "Misclassified" if label != y_test[idx] else "Correct"
            print(
                f"{status} - Predicted {label} with certainty {round(probabilities[idx], 2)} for document: {doc_titles[idx]}")

        return X_test, y_test, y_pred, doc_titles

    @staticmethod
    def reduce_dimentionality(X, y, feature, ngrams, max_features, max_cull):
        tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                                analyzer=feature,
                                ngram_range=ngrams,
                                max_features=max_features,
                                max_df=max_cull)
        scaler = StandardScaler(with_mean=False)
        svd = TruncatedSVD(n_components=2)
        model = SVC(kernel='linear', C=100)
        # Train data
        X_tfidf = tfidf.fit_transform(X)
        X = scaler.fit_transform(X_tfidf)
        X_svd = svd.fit_transform(X)
        feature_names = tfidf.get_feature_names_out()

        model = model.fit(X_svd, y)

        return model, feature_names, X_svd, y

    def get_feature_importance(self, X, y, **params):

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

    def viz_decision_boundaries(self, X, y, save_path, document_names, **params):

        model, _, X_train, y_train = self.reduce_dimentionality(
            X, y, **params)

        # Plot the data points
        plt.scatter(X_train[:, 0], X_train[:, 1], c=list(y_train),
                    cmap='viridis', edgecolor='k', s=40)

        # Label scattered dots with document names
        for i, name in enumerate(document_names):
            try:
                plt.text(X_train[i, 0], X_train[i, 1],
                         name, fontsize=8, ha='right')
            except IndexError:
                continue

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


class SVMxai(SVMModel):
    def __init__(self, df: pd.DataFrame, n: int, data_type: str):
        super().__init__(df)
        self.n = n
        self.data_type = data_type

    def get_explanations(self, feature, ngrams, max_features, max_cull):
        # Train data
        traindata = self.df[~self.df['title'].str.contains('#|VII')]
        X_train, y_train = traindata['text'], traindata['label']
        # Test data
        testdata = self.df[self.df['title'].str.contains('test')]
        X_test, y_test = testdata['text'], testdata['label']

        tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                                analyzer=feature,
                                ngram_range=ngrams,
                                max_features=max_features, max_df=max_cull)
        scaler = StandardScaler(with_mean=False)
        model = SVC(kernel='linear', C=100,
                    probability=True, random_state=54)

        X_train_tfidf = tfidf.fit_transform(X_train).toarray()
        X_train = scaler.fit_transform(X_train_tfidf)

        X_test_tfidf = tfidf.transform(X_test).toarray()
        X_test = scaler.fit_transform(X_test_tfidf)
        feature_names = tfidf.get_feature_names_out()
        cls = model.fit(X_train, y_train)

        self.evaluate_predictions(self.df, cls, X_test, y_test)

        shap_values, values = self.compute_shap_values(
            cls, X_train, X_test, feature_names)

        # Plots
        self.individual_predictions(
            doc_instance=shap_values[21], title="False Positive prediction for Pseudo Epinomis", data_type=self.data_type)
        # Absolute mean SHAP: see the features that significantly affect model predictions
        self.summary_plot(shap_values, feature_names, self.n, self.data_type)
        self.plot_feature_importance(
            values, feature_names, self.n, self.data_type)

    def evaluate_predictions(self, df, cls, X_test, y_test):
        y_pred = cls.predict(X_test)

        correct_pos = []
        correct_neg = []
        missed_pos = []
        pred_index = []
        i = 0
        for idx, pred in zip(y_test.index, y_pred):
            print(
                f"Index {idx} {df.title.loc[idx]}:\tpredicted {pred}, true: {df.label.loc[idx]}")
            if pred == df.label.loc[idx] and pred == 1:
                correct_pos.append(idx)
                pred_index.append(i)
            elif pred == df.label.loc[idx] and pred == 0:
                correct_neg.append(idx)
            elif pred != df.label.loc[idx] and df.label.loc[idx] == 1:
                missed_pos.append(idx)
            i += 1
        # print(y_pred.tolist())
        # print(pred_index)

    def compute_shap_values(self, cls, X_train, X_test, feature_names):
        # Standard SHAP values
        explainer = shap.Explainer(cls, X_train, feature_names=feature_names)
        shap_values = explainer(X_test)
        values = explainer.shap_values(X_test)
        return shap_values, values

    @staticmethod
    def summary_plot(shap_values, feature_names, n, data_type):
        plt.figure(figsize=(15, 10))
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            max_display=n,
            plot_type='violin',
            show=False)
        plt.title('Feature Value Summary',
                  loc='right', fontsize=20, weight='bold')
        plt.tight_layout()
        plt.savefig(f'{data_type}_summary.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

    @staticmethod
    def plot_feature_importance(values, feature_names, n, data_type):
        shap_df = pd.DataFrame(
            values, columns=feature_names).sort_index(axis=1)
        vals = np.abs(shap_df).mean(0)
        # Create DataFrame for feature importance
        feature_importance = pd.DataFrame(
            list(zip(feature_names.tolist(), vals)),
            columns=["feature", "importance"])

        feature_importance = feature_importance.set_index("feature")
        feature_importance = feature_importance.reindex(
            feature_importance['importance'].abs().sort_values(ascending=False).index)

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(15, 12))
        top_features = feature_importance.head(n)

        hbars = ax.barh(top_features.index, top_features.importance,
                        align='center')
        # ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance', fontsize=15)
        ax.set_title("Feature Importance", loc='center',
                     fontsize=20, weight='bold')
        # Label with specially formatted floats
        ax.bar_label(hbars, fmt='%.3f', fontsize=9)
        ax.set_xlim(right=0.03)  # adjust xlim to fit labels

        fig.savefig(f"{data_type}_feature_importance.png",
                    dpi=300)  # bbox_inches="tight",
        plt.close()

    @staticmethod
    def individual_predictions(doc_instance, title: str, data_type):
        """
        f(x) = predicted number in log odds
        E[f(x)] = avg prediction log odds
        Arrows: amount of that feature increasing/decreasing the prediction compared to the avg
        """
        plt.figure(figsize=(15, 10))
        shap.plots.waterfall(doc_instance, show=False)
        plt.title(title,
                  loc='right', fontsize=20, weight='bold')
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'{data_type}_waterfall.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
