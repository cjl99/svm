import pandas as pd
import numpy as np
from libsvm.svmutil import *

from cvxopt import matrix, solvers
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def svm_libsvm(X, y, svmtype: str = "hard", C: float = float('inf')):
    # convert ndarray to python array
    y = y.tolist()
    X = X.tolist()

    problem = svm_problem(y, X)

    # Set the parameters for the SVM model
    param = svm_parameter()
    param.kernel_type = 0  # Linear kernel
    if svmtype == "hard":
        param.C = float('inf')
    elif svmtype == "soft":
        param.C = C
    # else: default C=1

    # Train the SVM model
    model = svm_train(problem, param)


    # Get the support vector coefficients
    sv_coef = np.array(model.get_sv_coef())

    # Get the support vectors
    sv = np.array(model.get_SV())

    # Get the bias term
    bias = -model.rho[0]

    # Recover the hyperplane
    hyperplane = np.dot(sv.T, sv_coef)[:, 0]

    print("Hyperplane:", hyperplane)
    print("Bias:", bias)

    return model


def read_datasets(file_path: str):
    df = pd.read_csv(file_path)
    X = df.values[:, 3:]
    y = np.array([1 if i == "M" else -1 for i in df.values[:, 2]])

    assert X.shape[0] == y.shape[0], "X and y should have the same number of rows."
    return X, y

def primal_svm_qp(X, y, data):
    """This function solves the primal SVM problem using quadratic programming(cvxopt.solver.qp)."""
    n, d = X.shape[0], X.shape[1]

    P =  np.zeros((d+1, d+1))
    P[:d, :d] = np.eye(d)

    q = np.zeros((d+1, 1))

    # G = - np.hstack((y.reshape(-1, 1) * X, y.reshape(-1, 1)))  # dot product
    G = np.zeros((n, d + 1))
    G[:, :d] = -y.reshape(-1, 1) * X
    G[:, d] = -y
    h = np.array([-1.0] * n)
    # print(h.shape)
    # h = np.ones(n)* -1
    # print(h.shape)
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    w = np.array(sol['x'][:d]).flatten()
    b = np.array(sol['x'][d])

    # predict given data
    y_pred = np.sign(np.dot(data, w) + b)
    print(f"qp Predicted labels: {y_pred}")

if __name__ == "__main__":
    X, y = read_datasets("./data_a1/data.csv")
    n, d = X.shape[0], X.shape[1]
    print(f"There are {n} samples and {d} features in the dataset.")

    # use cvxopt.solver.qp
    num_test = 10
    test_data = np.random.random((num_test, 27))

    primal_svm_qp(X, y, test_data)
    
    
    
    
    model = svm_libsvm(X, y, "hard")

    # Create a new data point for prediction
    test_data = test_data.tolist()

    # libsvm predict data 
    predicted_label, p_acc, p_vals = svm_predict([-1] * num_test, test_data, model)

    print(f"libsvm Predicted labels: {predicted_label}")
    print(f"libsvm Accuracy: {p_acc}")
    print(f"libsvm Prediction values: {p_vals}")

