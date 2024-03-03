import pandas as pd
import numpy as np
import sys
import shap
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import Counter
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
        val_set = self.df[self.df['title'].str.contains('VII|val')]
        X_val, y_val, val_titles = val_set['text'], val_set['label'], val_set['title'].to_list(
        )

        # Train-test split, keeping class imbalance
        self.df = self.df[~self.df['title'].str.contains('VII|val')]
        X, y = self.df['text'], self.df['label']

        return X_val, y_val, val_titles, X, y

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
            "vect__ngram_range": [(3, 3), (4, 4), (5, 5), (5, 6), (4, 6)],
            # "vect__max_features": [1000, 3000],
            "clf__kernel": ["linear"]  # , "rbf"],
            # Not influent: "clf__class_weight": [None, 'balanced']
        }
        pipeline = Pipeline(
            [("vect", TfidfVectorizer(strip_accents=None, lowercase=False, max_features=1000,
                                      analyzer=feature, token_pattern=r"(?u)\b\w+(?:[-:]\w+)?(?:[-:]\w+)?\b")),
             ("clf", SVC(C=100, random_state=0))
             ])
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
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

    def perform_crossval(self, X, y, feature, ngrams, max_features, max_cull):
        tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                                analyzer=feature,
                                ngram_range=ngrams,
                                max_features=max_features, max_df=max_cull)
        scaler = StandardScaler(with_mean=False)

        X_train_tfidf = tfidf.fit_transform(X)
        X = scaler.fit_transform(X_train_tfidf)

        model = SVC(kernel='linear', C=100, probability=True, random_state=54)

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5)
        f1_scores = []

        for fold, (train, test) in enumerate(cv.split(X, y)):
            X_train_fold = X[train].toarray()
            y_train_fold = y.iloc[train]
            X_test_fold = X[test].toarray()
            y_test_fold = y.iloc[test]

            classifier = model.fit(X_train_fold, y_train_fold)
            y_pred = classifier.predict(X_test_fold)
            f1 = f1_score(y_test_fold, y_pred)  # Compute F1 score
            f1_scores.append(f1)

            print(f"Fold {fold + 1} F1 score: {f1}")

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        print(f"Mean F1 score: {mean_f1}, Std F1 score: {std_f1}")

    def execute_model(self, feature, ngrams, max_features, max_cull, validation=False):
        # Train-test split, keeping class imbalance
        traindata = self.df[~self.df['title'].str.contains('test')]
        X_train, y_train = traindata['text'], traindata['label']
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=3, stratify=y)
        if validation:
            X_train, y_train = self.df['text'], self.df['label']
            print(y_train.shape, Counter(y_train))

        pipe = Pipeline(
            [("vect", TfidfVectorizer(strip_accents=None, lowercase=False,
                                      analyzer=feature,
                                      ngram_range=ngrams,
                                      max_features=max_features,
                                      max_df=max_cull)),
             # token_pattern=r"(?u)\b\w+(?:[-:]\w+)?(?:[-:]\w+)?\b")),
             ("scalar", StandardScaler(with_mean=False)),
             ("clf", SVC(C=100, kernel='linear'))
             ])

        # Train the model
        model = pipe.fit(X_train, y_train)
        # Getting feature names
        feature_names = pipe.named_steps['vect'].get_feature_names_out()

        return model, X_train, y_train, feature_names

    def get_predictions(self, model, X_val=None, y_val=None, val_titles=None, validation=False):
        testdata = self.df[self.df['title'].str.contains('test')]
        X_test, y_test = testdata['text'], testdata['label']
        doc_titles = testdata['title'].to_list()

        if validation:
            X_test, y_test, doc_titles = X_val, y_val, val_titles
            print(y_test.shape, doc_titles)

        # Make y_pred on the test set
        y_pred = model.predict(X_test)
        probabilities = model.decision_function(X_test)

        # Evaluate the model
        print("Confusion Matrix:\n", confusion_matrix(
            y_test, y_pred, normalize='true'))

        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Print ground truth and predicted authors for misclassified documents
        y_test.reset_index(drop=True, inplace=True)
        for idx, label in enumerate(y_pred):
            status = "Misclassified" if label != y_test[idx] else "Correct"
            print(
                f"{status} - Predicted {label} with certainty {round(probabilities[idx], 2)} for document: {doc_titles[idx]}")

        return X_test, y_test, y_pred, probabilities, doc_titles

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

    def validation_predictions(self, cls, X_test, y_test):
        y_pred = cls.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def get_validation_xp(self, X_train, y_train, X_test, y_test, feature, ngrams, max_features, max_cull):

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

        self.validation_predictions(cls, X_test, y_test)

        shap_values, values = self.compute_shap_values(
            cls, X_train, X_test, feature_names)

        # Plots
        self.individual_predictions(
            doc_instance=shap_values[0], title="Prediction for VII Letter", data_type=self.data_type)
        # Absolute mean SHAP: see the features that significantly affect model predictions
        self.summary_plot(shap_values, feature_names, self.n, self.data_type)
        self.plot_feature_importance(
            values, feature_names, self.n, self.data_type)

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
        print("Classification Report:\n", classification_report(y_test, y_pred))

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
        # Set font size for x and y ticks
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance', fontsize=20)
        ax.set_title("Feature Importance", loc='center',
                     fontsize=40, weight='bold')
        # Label with specially formatted floats
        ax.bar_label(hbars, fmt='%.3f', fontsize=15)
        ax.set_xlim(right=0.03)  # adjust xlim to fit labels
        print(top_features.index)
        fig.savefig(f"SVM{data_type}ValFeatures.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def individual_predictions(doc_instance, title: str, data_type):
        """
        f(x) = predicted number in log odds
        E[f(x)] = avg prediction log odds
        Arrows: amount of that feature increasing/decreasing the prediction compared to the avg
        """
        plt.figure(figsize=(15, 10))
        shap.plots.waterfall(doc_instance, max_display=21, show=False)
        plt.title(title,
                  loc='left', fontsize=20, weight='bold')
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'{data_type}_waterfall.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
