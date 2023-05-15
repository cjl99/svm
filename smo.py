import numpy as np

def InteriorPointSolver:
    """Interior point method for quadratic programming."""
    def solve():
        import cvxopt
        import cvxopt.solvers
        solution = cvxopt.solvers.qp(cvxopt.matrix(Q), cvxopt.matrix(p), cvxopt.matrix(G), cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
        return np.array(solution['x'])


class QPSolver:
    """Super Class for quadratic programming solver."""
    def __init__(self, Q, p, G, h, A, b):
        self.Q = Q
        self.p = p
        self.G = G
        self.h = h
        self.A = A
        self.b = b
    


class SMO:
    def __init__(self, X, y, C, tol, max_iter):
        self.X = X
        self.y = y
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = np.zeros(len(X))
        self.b = 0

    def train(self):
        num_samples, num_features = self.X.shape
        iter_count = 0
        while iter_count < self.max_iter:
            num_changed_alphas = 0
            for i in range(num_samples):
                error_i = self.calculate_error(i)
                if (self.y[i] * error_i < -self.tol and self.alpha[i] < self.C) or \
                   (self.y[i] * error_i > self.tol and self.alpha[i] > 0):
                    j = self.select_second_alpha(i)
                    error_j = self.calculate_error(j)
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue
                    eta = 2.0 * self.X[i, :].dot(self.X[j, :].T) - \
                          self.X[i, :].dot(self.X[i, :].T) - \
                          self.X[j, :].dot(self.X[j, :].T)
                    if eta >= 0:
                        continue
                    self.alpha[j] -= self.y[j] * (error_i - error_j) / eta
                    self.alpha[j] = self.clip_alpha(self.alpha[j], H, L)
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
                    self.b = self.calculate_b(i, j, alpha_i_old, alpha_j_old)
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                iter_count += 1
            else:
                iter_count = 0

    def calculate_error(self, i):
        f_i = self.predict(self.X[i])
        return f_i - self.y[i]

    def select_second_alpha(self, i):
        num_samples = len(self.X)
        j = i
        while j == i:
            j = np.random.randint(0, num_samples)
        return j

    def clip_alpha(self, alpha, H, L):
        if alpha > H:
            alpha = H
        elif alpha < L:
            alpha = L
        return alpha

    def calculate_b(self, i, j, alpha_i_old, alpha_j_old):
        b1 = self.b - self.calculate_error(i) - self.y[i] * (self.alpha[i] - alpha_i_old) * self.X[i, :].dot(self.X[i, :].T) - self.y[j] * (self.alpha[j] - alpha_j_old) * self.X[i, :].dot(self.X[j, :].T)
        b2 = self.b - self.calculate_error(j) - self.y[i] * (self.alpha[i] - alpha_i_old) * self.X[i, :].dot(self.X[j, :].T) - self.y[j] * (self.alpha[j] - alpha_j_old) * self.X[j, :].dot(self.X[j, :].T)
        if 0 < self.alpha[i] < self.C:
            b = b1
        elif 0 < self.alpha[j] < self.C:
            b = b2
        else:
            b = (b1 + b2) / 2
        return b


    def predict(self, x):
        f = self.b
        for i in range(len(self.X)):
            f += self.alpha[i] * self.y[i] * self.X[i, :].dot(x.T)
        return np.sign(f)

if __name__ == "__main__":
    # 定义训练数据集
    X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
    y = np.array([-1, -1, 1, 1, -1])

    # 创建SMO对象并进行训练
    smo = SMO(X, y, C=1.0, tol=0.01, max_iter=100)
    smo.train()

    # 预测新样本
    new_sample = np.array([4, 4])
    prediction = smo.predict(new_sample)
    print("预测结果:", prediction)