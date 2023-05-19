import numpy as np
from cvxopt import matrix, solvers


class SVMModel():
    def __init__(self) -> None:
        # bias is the last term of w.
        self.w = None  # shape (d+1, ) 

    def fit(self, X, y):
        pass

    def predict(self, X):
        """Predict given data, which can be used for all SVM Model."""
        n = X.shape[0]
        y_values = np.concatenate((X, np.ones((n, 1))), axis=1) @ self.w  # (n, d+1) @ (d+1, ) = (n, )
        y_labels = np.sign(y_values)  # (n, )
        
        return y_values, y_labels

class Primal_Hard_SVM(SVMModel):
    """Primal Hard-margin SVM problem solved by cvxopt.solvers.qp."""
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X, y):
        n, d = X.shape[0], X.shape[1]

        P =  np.zeros((d+1, d+1)) # shape (d+1, d+1)
        P[:d, :d] = np.eye(d)

        q = np.zeros((d+1, ))  # shape (d+1, )

        G = np.zeros((n, d + 1))  # shape (n, d+1)
        G[:, :d] = -y.reshape(-1, 1) * X # broadcast & pointwise product
        G[:, d] = -y 
        h = np.array([-1.0] * n)  # (n,)
        
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        self.w = np.array(sol['x']).flatten()

        return self


class Dual_Hard_SVM(SVMModel):
    """Primal Soft-margin SVM problem solved by cvxopt.solvers.qp."""
    def __init__(self) -> None:
        super().__init__()
        self.alpha = None  # shape (n, )
        self.w = None
    
    def fit(self, X, y):
        n, d = X.shape[0], X.shape[1]

        P = np.zeros((n, n)) # shape (n, n)
        for i in range(n):
            for j in range(n):
                P[i, j] = y[i] * y[j] * X[i] @ X[j].reshape(-1, 1)  # (1, d) @ (d, 1) = (1, 1)
        
        q = -np.ones((n, ))  # shape (n, )
        G = -np.eye(n)  # shape (n, n)
        h = np.zeros((n, ))  # shape (n, )
        A = y.reshape(1, -1)  # shape (1, n)
        b = 0.

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
        self.alpha = np.array(sol['x']).flatten()

        self.w = np.zeros((d, ))
        for i in range(n):
            self.w += self.alpha[i] * y[i] * X[i] # (1, ) * (1, ) * (1, d) = (d, ) 

        cnt = 0
        for i in range(n):
            if self.alpha[i] > 1e-9:
                b = y[i] - X[i] @ self.w
                cnt += 1
        b /= cnt
        self.w = np.concatenate((self.w, np.array([b])), axis=0)  # (d, ) -> (d+1, )

        return self


class Primal_Soft_SVM(SVMModel):
    """Primal soft-margin svm problem solved by cvxopt.solvers.qp."""
    def __init__(self, C=1.) -> None:
        super().__init__()
        self.C = C
    
    def fit(self, X, y):
        n, d = X.shape[0], X.shape[1]

        P = np.zeros((n+d+1, n+d+1))  # shape (n+d+1, n+d+1)
        P[:d, :d] = np.eye(d)  
        q = np.zeros((n+d+1, ))  # shape (n+d+1, )
        q[d+1:] = self.C * np.ones((n, )) # shape (n, )

        G = np.zeros((2*n, n+d+1))  # shape (2*n, n+d+1)
        G[:n, :d] = -y.reshape(-1, 1) * X # broadcast & pointwise product
        G[:n, d] = -y 
        G[:n, d+1:] = -np.eye(n)
        G[n:, d+1:] = -np.eye(n)

        h = np.zeros((2*n, ))  # shape (2*n, )
        h[:n] = -1.0 

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        self.w = np.array(sol['x']).flatten()[:d+1]  
        self.xi = np.array(sol['x']).flatten()[d+1:]  

        return self
    


class Dual_Soft_SVM(Primal_Soft_SVM):
    """Dual soft-margin svm problem solved by cvxopt.solvers.qp."""
    def __init__(self, C=1.) -> None:
        super().__init__(C)
        self.alpha = None  # shape (n, )
    
    def fit(self, X, y):
        n, d = X.shape

        P = np.zeros((n, n), dtype=float)  # shape (n, n)
        for i in range(n):
            for j in range(n):
                P[i, j] = y[i] * y[j] * X[i] @ X[j].reshape(-1, 1)  # (1, d) @ (d, 1) = (1, 1)
        
        q = -1 * np.ones((n, ))  # shape (n, )
    
        G = np.concatenate((-np.eye(n), np.eye(n)), axis=0)  # shape (2*n, n)
        h = np.concatenate((np.zeros((n, )), self.C * np.ones((n, ))), axis=0)  # shape (2*n, )
        A = y.reshape(1, -1)  # shape (1, n)
        b = 0.

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
        self.alpha = np.array(sol['x']).flatten()
        
        self.w = np.zeros((d, ))
        for i in range(n):
            self.w += self.alpha[i] * y[i] * X[i] # (1, ) * (1, ) * (1, d) = (d, ) 

        cnt = 0
        for i in range(n):
            if self.alpha[i] > 1e-8:
                cnt += 1
                b += y[i] - X[i] @ self.w
        b /= cnt

        self.w = np.concatenate((self.w, np.array([b])), axis=0)  # (d, ) -> (d+1, )

        return self

class Addition_Primal(Primal_Soft_SVM):
    def __init__(self, C) -> None:
        super().__init__(C)

    def fit(self, X, y):
        n, d = X.shape[0], X.shape[1]

        P = np.zeros((n+d+1, n+d+1))  # shape (n+d+1, n+d+1)
        P[:d, :d] = np.eye(d)  
        q = np.zeros((n+d+1, ))  # shape (n+d+1, )
        # modify here 
        q[d+1:] = self.C / n * np.ones((n, )) 

        G = np.zeros((2*n, n+d+1))  # shape (2*n, n+d+1)
        G[:n, :d] = -y.reshape(-1, 1) * X # broadcast & pointwise product
        G[:n, d] = y  # here is different 
        G[:n, d+1:] = -np.eye(n)
        G[n:, d+1:] = -np.eye(n)

        h = np.zeros((2*n, ))  # shape (2*n, )
        h[:n] = -1.0 

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        self.w = np.array(sol['x']).flatten()[:d+1]  
        self.xi = np.array(sol['x']).flatten()[d+1:]  

        return self    

    def predict(self, X):
        """Predict given data, which can be used for all SVM Model."""
        n = X.shape[0]
        y_values = np.concatenate((X, np.ones((n, 1))*-1), axis=1) @ self.w  # (n, d+1) @ (d+1, ) = (n, )
        y_labels = np.sign(y_values)  # (n, )
        
        return y_values, y_labels

class Addition_Dual(Primal_Soft_SVM):
    def __init__(self, C) -> None:
        super().__init__(C)
    
    def fit(self, X, y):
        n, d = X.shape

        P = np.zeros((n, n), dtype=float)  # shape (n, n)
        for i in range(n):
            for j in range(n):
                P[i, j] = y[i] * y[j] * X[i] @ X[j].reshape(-1, 1)  # (1, d) @ (d, 1) = (1, 1)
        
        q = -1 * np.ones((n, ))  # shape (n, )
    
        G = np.concatenate((-np.eye(n), np.eye(n)), axis=0)  # shape (2*n, n)
        h = np.concatenate((np.zeros((n, )), self.C / n * np.ones((n, ))), axis=0)  # modify here 
        A = y.reshape(1, -1)  # shape (1, n)
        b = 0.

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
        self.alpha = np.array(sol['x']).flatten()
        
        self.w = np.zeros((d, ))
        for i in range(n):
            self.w += self.alpha[i] * y[i] * X[i] # (1, ) * (1, ) * (1, d) = (d, ) 

        cnt = 0
        for i in range(n):
            if self.alpha[i] > 1e-8:
                cnt += 1
                b += X[i] @ self.w - y[i]
        b /= cnt

        self.w = np.concatenate((self.w, np.array([b])), axis=0)  # (d, ) -> (d+1, )

        return self
    
    def predict(self, X):
        """Predict given data, which can be used for all SVM Model."""
        n = X.shape[0]
        y_values = np.concatenate((X, np.ones((n, 1))*-1), axis=1) @ self.w  # (n, d+1) @ (d+1, ) = (n, )
        y_labels = np.sign(y_values)  # (n, )
        
        return y_values, y_labels