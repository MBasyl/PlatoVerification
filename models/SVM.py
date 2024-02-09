import pandas as pd
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout = open(f'SVM_log.txt', 'a')
f = "tinydata/PLAINdata.csv"
df1 = pd.read_csv(f)
fother = "tinydata/PLAINDataotherAuthors.csv"
df2 = pd.read_csv(fother)

df = pd.concat([df1, df2], ignore_index=True, axis=0)

print(f"\n\n#################\n{f}")
# print("Balance: ", df.label.value_counts())

df.rename(columns={'label': 'multiclass_label'}, inplace=True)
df['binary_label'] = df['multiclass_label'].apply(
    lambda x: 1 if x in [7, 9] else 0)
# lambda x: 1 if x == 7 else 0)

testdata = df[df['author'].str.contains('test')]
X_test = testdata['text']
y_test = testdata['binary_label']

traindata = df[~df['author'].str.contains('test')]
# traindata = traindata[~traindata['author'].str.contains('#|Ari')]

X_train = traindata['text']
y_train = traindata['binary_label']

print("Overview number authors:", len(set(traindata.author.tolist())))
print("Train Authors", set(traindata.author.tolist()))
print("Test Authors", set(testdata.author.tolist()))
print("Shapes: ", X_test.shape, X_train.shape)
print("Balance: ", y_train.value_counts())

# exit(0)
# Create a pipeline with a CountVectorizer and SVM classifier
model = make_pipeline(
    # Convert a collection of text documents to a matrix of token counts
    TfidfVectorizer(strip_accents=None, lowercase=False,
                    analyzer='char', ngram_range=(3, 9), max_features=3000),
    # Standardize features by removing the mean and scaling to unit variance
    StandardScaler(with_mean=False),
    SVC(kernel='rbf', C=100)    # Linear SVM classifier try kai rbf
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
# model.predict_proba() !!!!!!

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Print ground truth and predicted authors for misclassified documents
misclassified_indices = np.where(predictions != y_test)[0]
right_classification = np.where(predictions == y_test)[0]

print("\nMisclassified Documents:")
for idx in misclassified_indices:
    print(
        f"Predicted {predictions[idx]} for Author: {testdata['author'].iloc[idx]}")  # {testdata['title'].iloc[idx]}")

print("Correct Documents:")
for idx in right_classification:
    print(
        f"Predicted {predictions[idx]} for Author: {testdata['author'].iloc[idx]}")  # , {testdata['title'].iloc[idx]}")

print("#################")
