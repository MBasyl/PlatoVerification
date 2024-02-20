import numpy as np
import shap
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def summary_plot(data, shap_values, feature_names, n):
    plt.figure(figsize=(15, 10))
    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        max_display=n,
        plot_type='violin')
    plt.title('Feature Value Summary',
              loc='right', fontsize=20, weight='bold')
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{data}summary.png',
                bbox_inches='tight', dpi=300)


def plot_feature_importance(data, values, feature_names, n=10):
    shap_df = pd.DataFrame(values, columns=feature_names).sort_index(axis=1)
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

    fig.savefig(f"{data}feature_importance.png",
                dpi=300)  # bbox_inches="tight",


def individual_predictions(data, doc_instance, title: str):
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
    plt.savefig(f'{data}waterfallplaLaw.png',
                bbox_inches='tight', dpi=300)


data = 'PLAIN'
feature = 'char'
ngram = (4, 6)
n = 15
csv_data = f"{data}dataset_obfuscate.csv"

df = pd.read_csv(csv_data)
val_set = df[df['title'].str.contains('VII|#')]

# Train-test split, keeping class imbalance
df = df[~df['title'].str.contains('VII|#')]
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3, stratify=y)

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False,
                        analyzer=feature,
                        ngram_range=ngram,  # 4,6 char
                        max_features=1000, max_df=0.8)  # 0.8 char
scaler = StandardScaler(with_mean=False)
model = SVC(kernel='linear', C=100,
            probability=True, random_state=54)

# Vectorize
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_train = scaler.fit_transform(X_train_tfidf)
X_test_tfidf = tfidf.fit_transform(X_test).toarray()
X_test = scaler.fit_transform(X_test_tfidf)
feature_names = tfidf.get_feature_names_out()

# Train and predict
cls = model.fit(X_train, y_train)
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
# exit(0)
# Standard SHAP values
explainer = shap.Explainer(model, X_train, feature_names=feature_names)
shap_values = explainer(X_test)
values = explainer.shap_values(X_test)
n = 15
# Plots
individual_predictions(
    data, doc_instance=shap_values[8], title="False Negative prediction for Plato Law")
# 

# Absolute mean SHAP
# see the features that significantly affect model predictions
# plot_feature_importance(data, values, feature_names, n)
# summary_plot(data, shap_values, feature_names, n)
