import numpy as np

epsilon = 0.000000001
def main():
    def f(x):
        return np.exp(-x**2)-2*x

    def get_error(al, bl):
        return np.absolute(al - bl)

    def is_within_error(al, bl):
        return get_error(al, bl) < epsilon


    i = 0
    a = 0 # pasirenkame intervala, kuriame bus sprendinys pagal grafika graph-1.png
    b = 1
    a_sign = f(a) > 0
    b_sign = f(b) > 0
    if b_sign == a_sign:
        return "error"
    print(i, ", [", a, ",", b, "] ,", get_error(a, b))
    while not is_within_error(a, b):
        i += 1
        error = get_error(a, b)
        c = (b + a) / 2
        c_val = f(c)
        c_sign = c_val > 0
        if c_sign == a_sign:
            a = c
        elif c_sign == b_sign:
            b = c
        print(i, ", [", a, ",", b, "] ,", error)
        if c_val == 0:
            return c
    return b

print("RESULT", main())