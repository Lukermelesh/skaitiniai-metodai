import numpy as np
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


np.seterr('raise')
n = 160
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
        print(x)
        print("Total time:", end - start)
        break
    else:
        prev_x = x