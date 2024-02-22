"""
Script modified from R. Layton 
"""
from scipy.spatial import distance
from operator import itemgetter
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import *
import numpy as np
from operator import itemgetter
from collections import defaultdict
import numpy as np


class RLP(BaseEstimator, ClassifierMixin):
    def __init__(self, n, L, char=True):

        self.n = n
        self.L = L
        self.char = char
        self.language_profile = None

    def predict(self, documents):
        # Predict which of the authors wrote each of the documents
        predictions = np.array([self.predict_single(document)
                               for document in documents])
        return predictions

    def predict_single(self, document):
        # Predicts the author of a single document
        # Profile of current document
        profile = self.create_profile(document)
        # Distance from document to each author
        distances = [(author, self.compare_profiles(profile, self.author_profiles[author]))
                     for author in self.author_profiles]
        # Get the nearest pair, and the author from that pair
        prediction = sorted(distances, key=itemgetter(1))[0][0]
        return prediction

    def predict_distance(self, documents):
        """MY OWN ADDITION TO SET CERTAINTY THRESHOLD"""
        predictions = [self.predict_dist_single(document)
                       for document in documents]
        # print(predictions)
        return predictions

    def predict_dist_single(self, document):
        # Predicts the author of a single document
        # Profile of current document
        profile = self.create_profile(document)
        # Distance from document to each author
        distances = [(author, self.compare_profiles(profile, self.author_profiles[author]))
                     for author in self.author_profiles]

        # Get the nearest pair, and the author from that pair
        # predictions = sorted(distances, key=itemgetter(1))[:3]

        label_predicted = distances[0][0]
        # round up distances for easier read
        smaller_distance = distances[0][1]
        larger_distance = distances[1][1]
        distance_gap = larger_distance - smaller_distance
        # pred1 = predictions[0][1]
        # pred2 = predictions[1][1]
        # probas = []
        # if pred1 < 0.55 or pred2-pred1 > 0.12:
        #     probas.append((predictions[0][0], pred1))
        # else:
        #     probas.append(("NA", pred1))
        # return probas
        return (label_predicted, round(smaller_distance, 3), round(distance_gap, 3))

    def compare_profiles(self, profile1, profile2):
        # All n-grams in the top L of either profile.
        ngrams = list(self.top_L(profile1).keys()) + \
            list(self.top_L(profile1).keys())
        # Profile vector for profile 1
        d1 = np.array([profile1.get(ng, 0.) for ng in set(ngrams)])
        # Profile vector for profile 2
        d2 = np.array([profile2.get(ng, 0.) for ng in set(ngrams)])
        return distance.cosine(d1, d2)

    def top_L(self, profile):
        threshold = sorted(map(abs, profile.values()))[-self.L]
        copy = defaultdict(float)
        for key in profile:
            if abs(profile[key]) >= threshold:
                copy[key] = profile[key]
        return copy

    def fit(self, documents, classes):
        self.language_profile = self.create_profile(documents)
        # Fits the current model to the training data provided
        # Separate documents into the sets written by each author
        author_documents = ((author, [documents[i]
                                      for i in range(len(documents))
                                      if classes[i] == author])
                            for author in set(classes))
        # Profile each of the authors independently
        self.author_profiles = {author: self.create_profile(cur_docs)
                                for author, cur_docs in author_documents}

    def create_profile(self, documents):
        # Creates a profile of a document or list of documents.
        if isinstance(documents, str):
            documents = [documents,]
        # profile each document independently
        if self.char:
            # print("Counting characters...")
            # BETTER TO NOT NORMALIZE!!
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
