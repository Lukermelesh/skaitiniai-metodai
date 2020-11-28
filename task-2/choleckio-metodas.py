import numpy as np

def create_matrix(fill_with, n):
    matrix = np.zeros((n, n))
    print(matrix)
    for i in range(0, len(fill_with)):
        val = fill_with[i:i+1]
        matrix += np.diag(val * (n - i), i) + np.diag(val * (n - i), -i) if i > 0 else np.diag(val * n)
    return matrix

n = 10
c = 1 / (n + 1)**2

x0 = np.array([1]*n)

matrix = create_matrix([30, -16, 1], n)
print(matrix)

a = np.linalg.solve(matrix, x0)
print(a)