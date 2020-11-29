import numpy as np
import math
import time

def create_matrix(fill_with, n):
    matrix = np.zeros((n, n))
    for i in range(0, len(fill_with)):
        val = fill_with[i:i+1]
        matrix += np.diag(val * (n - i), i) + np.diag(val * (n - i), -i) if i > 0 else np.diag(val * n)
    return matrix

def cholesky(A):
    n = len(A)
    L = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if j < i:
                sum = 0
                for k in range(0, j):
                    sum += L[i, k] * L[j, k]
                L[i, j] = (A[i, j] - sum) / L[j, j]
            elif i == j:
                sum = 0
                for k in range(0, i):
                    sum += L[i, k]**2
                L[i, i] = math.sqrt(A[i, i] - sum)
    return L

def solve(matrix, B, is_upper):
    n = len(B)
    res = np.array([0.]*n)
    if is_upper:
        res[n - 1] = B[n - 1] / matrix[n - 1][n - 1]
        for i in range(n - 2, -1, -1):
            temp = 0
            for j in range(i, n):
                temp += matrix[i][j] * res[j]
            res[i] = (B[i] - temp) / matrix[i][i]
    else:
        res[0] = B[0] / matrix[0][0]
        for i in range(1, n):
            temp = 0
            for j in range(0, i):
                temp += matrix[i][j] * res[j]
            res[i] = (B[i] - temp) / matrix[i][i]
    return res

def get_error(x1, x2):
    x = np.subtract(x1, x2)
    return abs(max(x.min(), x.max(), key=abs))

def f(x, c):
    n = len(x)
    res = np.array([0.]*n)
    for i in range(0, n):
        res[i] = c + 2*((x[i + 1] if i + 1 < n else 0) - (x[i - 1] if i > 0 else 0))**2
    return res

np.seterr('raise')
n = 30
c = 1 / (n + 1)**2
epsilon = 0.01
start = time.time()

prev_x = np.array([.5]*n) # x0 pasirenkame vektoriu
A = create_matrix([30, -16, 1], n)
# npres = np.linalg.solve(A, prev_x)
# print(npres) #pasitikrinimui
cholesky_start = time.time()
L = cholesky(A)
cholesky_end = time.time()
print("Cholesky decomposition took:", cholesky_end - cholesky_start)
while True:
    Y = solve(L, prev_x, False)
    x = solve(np.transpose(L), Y, True)
    error = get_error(x, prev_x)
    # print("ERROR", error)
    if error <= epsilon:
        print(x)
        break
    else:
        prev_x = f(x, c)
        # print("prev_x", prev_x)

end = time.time()
print("Total time:", end - start)