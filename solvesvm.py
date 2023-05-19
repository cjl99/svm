import os
import numpy as np
import pandas as pd
from models import Primal_Hard_SVM, Dual_Hard_SVM, Primal_Soft_SVM, Dual_Soft_SVM, Addition_Primal, Addition_Dual
from libsvm.svmutil import *
import argparse


def compare_weigts(primal, dual, libsvm):
    "Compute L2 distance between primal, dual and libsvm weights."
    assert primal.shape == dual.shape == libsvm.shape, "weights should have the same shape."
    primal_dist = np.linalg.norm(primal - libsvm)
    dual_dist = np.linalg.norm(dual - libsvm)
    primal_dual = np.linalg.norm(primal - dual)

    print(f"distance between primal weights and dual weights: {primal_dual} ")
    print(f"distance between primal weights and libsvm weights: {primal_dist} ")
    print(f"distance between dual weights and libsvm weights: {dual_dist}")


def read_datasets(file_path: str):
    df = pd.read_csv(file_path)
    X = df.values[:, 3:].astype(float)
    y = np.array([1 if i == "M" else -1 for i in df.values[:, 2]]).astype(float)

    assert X.shape[0] == y.shape[0], "X and y should have the same number of rows."
    return X, y
    
def split_data(X, y):
    ratio = 0.7
    n = X.shape[0]
    n_train = int(n * ratio)

    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]

def calcuate_acc(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
    

def run_easy_hard_test():
    X = np.array([[3,3], [4,3], [1,1]], dtype=float)
    y = np.array([1, 1, -1], dtype=float)
    test_data = np.array([[1, 1], [2, 3], [3, 9]])
    y_true = np.array([-1, 1, 1], dtype=float)

    # primal hard-margin svm
    my_sol = Primal_Hard_SVM().fit(X, y)
    print("primal hard-margin svm coefs: ", my_sol.w)
    y_value, y_pred = my_sol.predict(test_data)
    print(f"primal hard-margin svm acc: {calcuate_acc(y_true, y_pred)}")

    # dual hard-margin svm
    my_sol = Dual_Hard_SVM().fit(X, y)
    print("dual hard-margin svm coefs:", my_sol.w)
    y_value, y_pred = my_sol.predict(test_data)
    print(f"dual hard-margin svm acc: {calcuate_acc(y_true, y_pred)}")


def run_easy_soft_test():
    # 4 points are linearly separable, 1 points are not
    X = np.array([[3,3], [4,3], [1,1], [0,0], [0,1], [1,0]], dtype=float)
    y = np.array([1, 1, -1, -1, 1, -1], dtype=float)
    
    test_data = np.array([[-1, -1], [2, 3], [3, 9]])
    y_true = np.array([-1, 1, 1], dtype=float)

    # primal soft-margin svm
    my_sol = Primal_Soft_SVM().fit(X, y)
    print("primal soft-margin svm coefs: ", my_sol.w)
    y_value, y_pred = my_sol.predict(test_data)
    print(f"primal soft-margin svm acc: {calcuate_acc(y_true, y_pred)}")

    # dual soft-margin svm
    my_sol = Dual_Soft_SVM().fit(X, y)
    print("dual soft-margin svm coefs:", my_sol.w)
    y_value, y_pred = my_sol.predict(test_data)
    print(f"dual soft-margin svm acc: {calcuate_acc(y_true, y_pred)}")


def run_data_a1_soft_test(C: float = 1.5):
    X, y = read_datasets("./data_a1/data.csv")
    n, d = X.shape
    print(f"There are {n} samples and {d} features in the dataset.")

    X, y, test_data, y_true = split_data(X, y)

    # primal soft-margin svm
    primal_soft_sol = Primal_Soft_SVM(C).fit(X, y)
    y_value, y_pred = primal_soft_sol.predict(test_data)
    primal_soft_acc = calcuate_acc(y_true, y_pred)

    # dual soft-margin svm
    dual_soft_sol = Dual_Soft_SVM(C).fit(X, y)
    y_value, y_pred = dual_soft_sol.predict(test_data)
    dual_soft_acc = calcuate_acc(y_true, y_pred)

    # libsvm
    prob = svm_problem(y.tolist(), X.tolist())
    param = svm_parameter('-s 0 -t 0')  # -s 0: C-SVC, -t 0: linear kernel
    param.C = C
    model = svm_train(prob, param)
    sv_coef = np.array(model.get_sv_coef()).reshape(-1,1)# (n_sv, 1)
    sv = model.get_SV()
    support_vecs = []
    # convert dict to array, ignore key
    for i, s in enumerate(sv):
        support_vecs.append(list(s.values()))
    sv = np.array(support_vecs, dtype=float)  # (n_sv, n_features)

    weights = np.sum(sv_coef * support_vecs, axis=0)  # (n_features, )
    b = -model.rho[0]  # get bias term
    libsvm_weights = np.concatenate((weights, [b]), axis=0)
    y_pred = svm_predict(y_true.tolist(), test_data.tolist(), model)[0]
    libsvm_acc = calcuate_acc(y_true, y_pred)

    print(f"====== results for C={C} ======")
    compare_weigts(primal_soft_sol.w, dual_soft_sol.w, libsvm_weights)

    print(f"primal soft-margin svm acc: {primal_soft_acc}")
    print(f"dual soft-margin svm acc: {dual_soft_acc}")
    print(f"libsvm acc: {libsvm_acc}")


def run_addition_test(C:float = 1.5):
    # easy test
    X, y = read_datasets("./data_a1/data.csv")
    n, d = X.shape
    print(f"There are {n} samples and {d} features in the dataset.")

    X, y, test_data, y_true = split_data(X, y)

    # primal soft-margin svm -- addition version
    addi_primal = Addition_Primal(C).fit(X, y)
    print("addition primal soft-margin svm coefs: ", addi_primal.w)
    y_value, y_pred = addi_primal.predict(test_data)
    print(f"addition primal soft-margin svm acc: {calcuate_acc(y_true, y_pred)}")

    # dual soft-margin svm -- addition version
    addi_dual = Addition_Dual(C).fit(X, y)
    print("addition dual soft-margin svm coefs:", addi_dual.w)
    y_value, y_pred = addi_dual.predict(test_data)
    print(f"addition dual soft-margin svm acc: {calcuate_acc(y_true, y_pred)}")

    compare_weigts(addi_primal.w, addi_dual.w, np.zeros((28,)))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--C", type=float, default=1.5)
    args = args.parse_args()

    run_easy_hard_test()
    run_easy_soft_test()

    run_data_a1_soft_test(C = args.C)

    run_addition_test(C = args.C) 



