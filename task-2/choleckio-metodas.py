import numpy as np
import math

def fill_matrix_symetric(matrix, fill_with):
    start_at = math.ceil((len(fill_with) - 1) / 2)
    matrix_size = len(matrix)
    fill_length = len(fill_with)
    fill_with = fill_with.reshape(1, len(fill_with))
    for i in range(0, matrix_size):
        if i < start_at:
            matrix[i:i+1, :i+start_at+1] = fill_with[:,start_at-i:]
        elif i > matrix_size - start_at - 1:
            from_end = i - (matrix_size - start_at - 1)
            matrix[i:i+1, i-start_at:] = fill_with[:,:fill_length-from_end]
        else:
            matrix[i:i+1, i-start_at:i-start_at+fill_length] = fill_with

    return matrix



matrix =  np.zeros((10, 10))
matrix = fill_matrix_symetric(matrix, np.array([1, -16, 30, -16, 1]))
print(matrix)
