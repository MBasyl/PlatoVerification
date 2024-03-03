import pandas as pd
from models import SVM as svm
from models import SVM as xai
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


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


def perform_svm(df, **params):
    svm_model = svm.SVMModel(df)
    _, _, _, X, y = svm_model.process_data(drop_history=True)

    # Apply best hyperparams and get mean results from Cross-Validation
    svm_model.perform_crossval(X, y, **params)  # save_path=f'{data}SVMcvROC',

    # Train and test model
    model, X_train, y_train, feature_names = svm_model.execute_model(**params)
    X_test, y_test, _, y_probs, doc_names = svm_model.get_predictions(model)

    # Inspect features: not very informative --> use SHAP
    # svm_model.get_feature_importance(X, y, **params)

    # Visualize decision boundaries
    # svm_model.viz_decision_boundaries(X_test, y_test,
    #                                   save_path=f'{data}svmDecisionBoundary.png',
    #                                   document_names=doc_names, **params)
    plot_roc_curve(y_test, y_probs, save_path=f'{data}svmROC.png')
    # Use SHAP module for explainable ML
    svm_xai = xai.SVMxai(df, n=15, data_type='PARSED')
    svm_xai.get_explanations(**params)


def perform_validation(df, **params):

    svm_model = svm.SVMModel(df)
    X_val, y_val, val_titles, X, y = svm_model.process_data(
        drop_history=True)

    # Train and validate model
    model, X_train, y_train, feature_names = svm_model.execute_model(
        **params, validation=True)
    csv_data = f"{data}dataset_validation.csv"
    val_set = pd.read_csv(csv_data, sep=";")
    X_val, y_val, val_titles = val_set['text'], val_set['label'], val_set['title'].to_list(
    )
    X_test, y_test, _, y_probs, doc_names = svm_model.get_predictions(
        model, X_val, y_val, val_titles, validation=True)
    # plot_roc_curve(y_test, y_probs, save_path=f'{data}svmROC.png')
    svm_xai = xai.SVMxai(df, n=15, data_type='PLAIN')
    svm_xai.get_validation_xp(X_train, y_train, X_test, y_test, **params)


if __name__ == "__main__":
    data = 'PARSED'
    csv_data = f"{data}personal_obfuscate.csv"
    df = pd.read_csv(csv_data, sep=";")

    # Dataset balance [94, 26]/[24, 6]
    params = {
        'feature': 'word',
        'ngrams': (3, 3),
        'max_features': 1000,
        'max_cull': 0.8}

    print(data, params)
    svm_model = svm.SVMModel(df)

    # Gridsearch for best hyperparameters
    # _, _, _, X, y = svm_model.process_data(drop_history=True)
    # svm_model.perform_SearchGrid(
    #     X, y, feature='word', save_path=f'{data}gridSVM')

    # perform_svm(df, **params)
    # PARSED: {'clf__kernel': 'linear', 'vect__max_df': 0.4, 'vect__ngram_range': (4, 4)}
    # {'clf__kernel': 'linear', 'vect__max_df': 0.8, 'vect__ngram_range': (3, 3)}
    # PLAIN: {'clf__kernel': 'linear', 'vect__max_df': 0.8,'vect__ngram_range': (3, 6)
    # Alt: {'clf__kernel': 'linear', 'vect__max_df': 0.8, 'vect__ngram_range': (3, 6)}

    perform_validation(df, **params)
