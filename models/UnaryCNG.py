"""
Script based on R. Layton, GitHub 
"""
from scipy.spatial import distance
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import *
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd


class UnaryCNG(BaseEstimator, ClassifierMixin):
    def __init__(self, n, L, char=True):
        self.n = n
        self.L = L
        self.char = char
        self.language_profile = None
        self.target_profile = None
        self.threshold = None

    def predict(self, documents, measure='cosine'):
        # assert measure in ['cosine', 'minmax', 'cole']
        # Predict which of the authors wrote each of the documents
        all_distances = np.array([self.predict_distance(document)
                                  for document in documents])

        positive_threshold = self.threshold[1]+self.threshold[2]

        # Make predictions based on threshold
        if measure == 'cosine':
            cosine_threshold = self.threshold[1][0]+self.threshold[2][0]
            predictions = [0 if dist[0] >
                           cosine_threshold else 1 for dist in all_distances]
        elif measure == 'cole':
            cole_threshold = self.threshold[1][1]-self.threshold[2][1]
            predictions = [0 if dist[1] >
                           cole_threshold else 1 for dist in all_distances]
        else:
            raise ValueError(
                "Invalid measure. Please provide either 'cosine' or 'cole'.")

        # elif measure == 'cole':...
        return predictions, all_distances

    def predict_distance(self, document):
        # Distance from current document to target author
        distances = self.compare_profiles(
            self.target_profile, self.create_profile(document))
        return distances

    def compare_profiles(self, profile1: dict, profile2: dict):
        # All n-grams in the top L of either profile.
        ngrams = list(self.top_L(profile1).keys()) + \
            list(self.top_L(profile1).keys())
        # Profile vector for profile 1 (target)
        t = np.array([profile1.get(ng, 0.) for ng in set(ngrams)])
        # Profile vector for profile 2 (new instance)
        i = np.array([profile2.get(ng, 0.) for ng in set(ngrams)])
        # Cosine Distance
        cosine_dist = round(distance.cosine(t, i), 4)
        # Cole Correlation
        # Stack vectors vertically & USE ABSOLUTE VALUES?
        matrix = np.vstack((t, i))
        matrix = (matrix > 0).astype(int)  # Convert to boolean
        a = np.sum(np.logical_and(t, i))  # features present in both
        b = np.sum(np.logical_and(t, 1 - i))  # features present only in t
        c = np.sum(np.logical_and(1 - t, i))  # features present only in k
        p = matrix.shape[1]  # number of features in total
        d = p - (a + b + c)  # features absent in both
        correlation_dist = round((a * d - b * c) / ((a + b) * (b + d)), 4)
        # print(cosine_dist, correlation_dist)
        return cosine_dist, correlation_dist

    def top_L(self, profile: dict) -> dict:
        threshold = sorted(map(abs, profile.values()))[-self.L]
        copy = defaultdict(float)
        for key in profile:
            if abs(profile[key]) >= threshold:
                copy[key] = profile[key]
        return copy

    def fit(self, documents, classes):
        # Create baseline dataset language profile
        self.language_profile = self.create_profile(documents)

        # Profile each document independently: NEED THIS??
        # single_documents = []
        # for i in range(len(documents)):
        #     print(classes[i], documents[i])
        #     single_documents.append((classes[i], documents[i]))
        # self.document_profiles = {author: self.create_profile(cur_doc)
        #                           for author, cur_doc in single_documents}

        # Create a uniform PLATO (target) profile
        target_documents = [doc for doc, label in zip(
            documents, classes) if label == 1]  # <-- author_documents()
        self.target_profile = self.create_profile(target_documents)
        # Get threshold aka maximum distance for acceptable positive cur_doc
        thresholds = []
        # Iterate over each document_profile, if that document is in target_documents drop it from target_documents
        for i in range(len(target_documents)):
            target = target_documents[:i] + target_documents[i+1:]
            curr_doc = target_documents[i]
            # build a target_profile
            target_small_profile = self.create_profile(target)
            # Compare the target_profile minus the current document *** see Pliny work ****
            threshold = self.compare_profiles(
                target_small_profile, self.create_profile(curr_doc))
            thresholds.append(threshold)
        self.threshold = (np.median(thresholds, axis=0), np.mean(
            thresholds, axis=0), np.std(thresholds, axis=0))
        return self.threshold

    def create_profile(self, documents):
        # Creates a profile of a document or list of documents.
        if isinstance(documents, str):
            documents = [documents,]
        if isinstance(documents, np.ndarray):
            documents = documents.tolist()

        # profile each document independently
        if self.char:
            profiles = (count_ngrams(document, self.n, normalise=False)
                        for document in documents)
        else:
            profiles = (count_tokens(document, self.n, normalise=False)
                        for document in documents)
        # Merge the profiles
        main_profile = defaultdict(float)
        for profile in profiles:
            for ngram in profile:
                main_profile[ngram] += profile[ngram]
        # Normalise the profile
        num_ngrams = float(sum(main_profile.values()))
        for ngram in main_profile:
            main_profile[ngram] /= num_ngrams
        if self.language_profile is not None:
            # Recentre profile.
            for key in main_profile:
                main_profile[key] = main_profile.get(
                    key, 0) - self.language_profile.get(key, 0)
        # Note that the profile is returned in full, as exact frequencies are used
        # in comparing profiles (rather than chopped off)
        return main_profile

    def compute_feature_importance(self):
        target_profile = self.target_profile
        language_profile = self.language_profile

        # Compute feature importance by comparing the target author's profile with the language profile
        feature_importance = {}
        for ngram in target_profile:
            feature_importance[ngram] = abs(
                target_profile[ngram] - language_profile.get(ngram, 0))
        # Sort feature importance dictionary by values (importance scores)
        feature_df = pd.DataFrame(
            {'Coefficient': feature_importance.values(), 'Feature': feature_importance.keys()})
        feature_df = feature_df.reindex(
            feature_df['Coefficient'].abs().sort_values(ascending=False).index)

        return feature_df

    def plot_feature_importance(self, title, filename, top_n=15):
        feature_importance = self.compute_feature_importance()
        # Display the top N features
        top_features = feature_importance.head(top_n)

        # Create a bar plot
        plt.figure(figsize=(15, 12))
        plt.barh(top_features['Feature'],
                 top_features['Coefficient'], color='skyblue', align='center')
        plt.xlabel('Feature Importance')  # xticks rotation=90
        plt.title(title, weight='bold')
        plt.savefig(filename, bbox_inches='tight')

        plt.tight_layout()
        plt.show()
