import numpy as np
import math
import time

def create_matrix(fill_with, n):
    matrix = np.zeros((n, n))
    for i in range(0, len(fill_with)):
        val = fill_with[i:i+1]
        matrix += np.diag(val * (n - i), i) + np.diag(val * (n - i), -i) if i > 0 else np.diag(val * n)
    return matrix

def seidel(a, xOrg, b):
    x = xOrg.copy()
    n = len(x)
    for i in range(0, n):
        sum = 0
        for j in range(0, n):
            sum += a[i][j]*x[j] if j != i else 0
        x[i] = (b[i] - sum) / a[i][i]
    return x

def get_error(x1, x2):
    x = np.subtract(x1, x2)
    return abs(max(x.min(), x.max(), key=abs))

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

np.seterr('raise')
n = 1600
epsilon = 0.01
start = time.time()

prev_x = np.array([0.]*n) # x0 pasirenkame vektoriu
B = np.array([14] * n)
A = create_matrix([30, -10, 1], n)
# npres = np.linalg.solve(A, B)
# print(npres) #pasitikrinimui
while True:
    x = seidel(A, prev_x, B)
    error = get_error(x, prev_x)
    if error <= epsilon:
        end = time.time()
        # print(x)
        print("Total time:", end - start)
        break
    else:
        prev_x = x

start = time.time()
L = cholesky(A)
Y = solve(L, B, False)
x = solve(np.transpose(L), Y, True)
end = time.time()
print("Total Cholesky time:", end - start)
# print(x)