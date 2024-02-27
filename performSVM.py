import pandas as pd
from models import SVM as svm
from collections import Counter
from models import SVM as xai


def perform_svm(df, **params):
    svm_model = svm.SVMModel(df)
    X_val, y_val, X, y = svm_model.process_data(
        drop_history=True)

    # Apply best hyperparams and get mean results from Cross-Validation
    # svm_model.perform_crossval(X, y, save_path=f'{data}SVMcvROC', **params)

    # Train and test model
    model, _, _ = svm_model.execute_model(**params)
    X_test, y_test, _, doc_names = svm_model.get_predictions(model)

    # Inspect features: not very informative --> use SHAP
    # svm_model.get_feature_importance(X, y, **params)

    # Visualize decision boundaries
    svm_model.viz_decision_boundaries(X_test, y_test,
                                      save_path=f'{data}svmDecisionBoundary.png',
                                      document_names=doc_names, **params)


if __name__ == "__main__":
    data = 'Alt'
    csv_data = f"{data}dataset_obfuscate.csv"
    df = pd.read_csv(csv_data)

    # Dataset balance [94, 26]/[24, 6]
    params = {
        'feature': 'word',
        'ngrams': (3, 6),
        'max_features': 1000,
        'max_cull': 0.8}

    print(data, params)
    svm_model = svm.SVMModel(df)

    # Gridsearch for best hyperparameters
    # _, _, X, y = svm_model.process_data(drop_history=True)
    # svm_model.perform_SearchGrid(X, y, feature='word', save_path=f'{data}gridSVM')

    # perform_svm(df, **params)
    # PARSED: {'clf__kernel': 'linear', 'vect__max_df': 0.4, 'vect__ngram_range': (4, 4)}
    # PLAIN: {'clf__kernel': 'linear', 'vect__max_df': 0.8,'vect__ngram_range': (3, 6)
    # Alt: {'clf__kernel': 'linear', 'vect__max_df': 0.8, 'vect__ngram_range': (3, 6)}

    # Use SHAP module for explainable ML
    svm_xai = xai.SVMxai(df, n=15, data_type='Alt')
    svm_xai.get_explanations(**params)
