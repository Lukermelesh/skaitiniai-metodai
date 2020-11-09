import numpy as np

def g(x):
    return np.exp(-x**2)/2

epsilon = 0.000000001
q = 0.5 #pasirenkame pagal isvestines g(x)' grafika
def get_error(x, x_prev):
    return np.absolute(x - x_prev)

prev_x = 0
curr_x = 0
i = 0
while True:
    i += 1
    prev_x = curr_x
    curr_x = g(curr_x)
    error = get_error(curr_x, prev_x)
    print(i, ",", curr_x, ",", error)
    if error <= (1 - q) / q * epsilon:
        print("RESULT =", curr_x)
        break
