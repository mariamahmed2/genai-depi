import numpy as np
import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from mealpy.swarm_based.JA import LevyJA

# Function to preprocess and split the dataset
def before_models(method='BOW'):
    if method.lower() == 'bow':
        X = X_bow
    elif method.lower() == 'tfidf':
        X = X_tfidf
    else:
        raise ValueError('Enter a method from (bow, tfidf)')
        
    y = dataset['target']
    over = SMOTE()
    X_new, y_new = over.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, shuffle=True, stratify=y_new, test_size=0.2, random_state=15)
    
    return X_train, X_test, y_train, y_test

# SVC with Jaya optimizer
def svc_with_jaya(method, kernel_choices, C_range):
    X_train, X_test, y_train, y_test = before_models(method=method)
    KERNEL_ENCODER = LabelEncoder()
    KERNEL_ENCODER.fit(kernel_choices)

    def fitness_function(solution):
        kernel_encoded = int(solution[0])
        c = solution[1]
        kernel_decoded = KERNEL_ENCODER.inverse_transform([kernel_encoded])[0]
        svc = SVC(C=c, random_state=42, kernel=kernel_decoded)
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    problem = {"fit_func": fitness_function, "lb": [0, C_range[0]], "ub": [len(kernel_choices) - 1, C_range[1]], "minmax": "max"}
    model = LevyJA(epoch=15, pop_size=150)
    model.solve(problem)

    best_c = model.solution[0][1]
    best_kernel = KERNEL_ENCODER.inverse_transform([int(model.solution[0][0])])[0]
    svc = SVC(C=best_c, kernel=best_kernel)
    svc.fit(X_train, y_train)

    acc_train = round(100 * accuracy_score(y_train, svc.predict(X_train)), 4)
    acc_test = round(100 * accuracy_score(y_test, svc.predict(X_test)), 4)

    return svc, acc_train, acc_test, best_kernel, best_c

# KNN with Jaya optimizer
def knn_with_jaya(method, k_range, p_range):
    X_train, X_test, y_train, y_test = before_models(method=method)

    def fitness_function(solution):
        k = int(solution[0])
        p = int(solution[1])
        knn = KNeighborsClassifier(n_neighbors=k, p=p)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    problem = {"fit_func": fitness_function, "lb": [k_range[0], p_range[0]], "ub": [k_range[1], p_range[1]], "minmax": "max"}
    model = LevyJA(epoch=15, pop_size=150)
    model.solve(problem)

    best_k = int(model.solution[0][0])
    best_p = int(model.solution[0][1])
    knn = KNeighborsClassifier(n_neighbors=best_k, p=best_p)
    knn.fit(X_train, y_train)

    acc_train = round(100 * accuracy_score(y_train, knn.predict(X_train)), 4)
    acc_test = round(100 * accuracy_score(y_test, knn.predict(X_test)), 4)

    return knn, acc_train, acc_test, best_k, best_p

def main(method, model_type, kernel_choices, C_range, k_range, p_range):
    global X_bow, X_tfidf, dataset

    # Load dataset
    dataset = pd.read_excel('dataset_1_after_tokenized.xlsx')[['target', 'text']]

    # Extracting features using BOW
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=None, stop_words='english')
    X_bow = bow_vectorizer.fit_transform(dataset['text']).toarray()

    # Extracting features using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=None, stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(dataset['text']).toarray()

    dict_map = {0: 0, 4: 1, 2: 2}
    dataset['target'] = dataset['target'].map(dict_map)

    # Start MLflow run
    with mlflow.start_run():
        if model_type.lower() == 'svc':
            model, acc_train, acc_test, best_kernel, best_c = svc_with_jaya(method=method, kernel_choices=kernel_choices, C_range=C_range)
            mlflow.log_param("Model", "SVM with Jaya")
            mlflow.log_param("Best Kernel", best_kernel)
            mlflow.log_param("Best C", best_c)
        elif model_type.lower() == 'knn':
            model, acc_train, acc_test, best_k, best_p = knn_with_jaya(method=method, k_range=k_range, p_range=p_range)
            mlflow.log_param("Model", "KNN with Jaya")
            mlflow.log_param("Best n_neighbors", best_k)
            mlflow.log_param("Best P", best_p)
        else:
            raise ValueError("Model type not supported. Use 'svc' or 'knn'.")

        mlflow.log_param("Method", method)
        mlflow.log_metric("Train Accuracy", acc_train)
        mlflow.log_metric("Test Accuracy", acc_test)
        mlflow.sklearn.log_model(model, model_type.lower() + "_model")
        
        print(f"Train Accuracy: {acc_train}%")
        print(f"Test Accuracy: {acc_test}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models with Jaya optimization using MLflow.")
    parser.add_argument("--method", type=str, required=True, help="Feature extraction method (BOW or TFIDF).")
    parser.add_argument("--model", type=str, required=True, help="Model type (SVC or KNN).")
    parser.add_argument("--kernels", type=str, nargs='+', default=['linear', 'poly', 'rbf', 'sigmoid'], help="Kernels for SVM.")
    parser.add_argument("--C_range", type=float, nargs=2, default=[0.01, 700], help="Range for C parameter in SVM.")
    parser.add_argument("--k_range", type=int, nargs=2, default=[2, 15], help="Range for n_neighbors in KNN.")
    parser.add_argument("--p_range", type=int, nargs=2, default=[1, 4], help="Range for p parameter in KNN.")
    args = parser.parse_args()

    main(args.method, args.model, args.kernels, args.C_range, args.k_range, args.p_range)
