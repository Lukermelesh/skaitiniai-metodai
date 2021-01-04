import numpy as np

def simpson(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    sum_odd = 0
    for i in range(1, n - 1, 2):
        sum_odd += f(x[i])

    sum_even = 0
    for i in range(2, n - 2, 2):
        sum_even += f(x[i])

    print('error', round(error(240,a,b,n), 6))
    return ((b - a)/(3*n))*(f(a) + 4*sum_odd + 2*sum_even + f(b))

def function(x): return 2/x**2

def deriv(x): return 240/x**6

def error(d, a, b, n):
    h = (b - a) / n
    return d*((b - a)/180)*h**4

result = simpson(function, 1.0, 4.0, 320)
print('result', round(result, 4))
print('real_error', round(1.5 - result, 6))

# n     val     err       real_err
# 20    1.4585  0.002025  0.0415
# 40    1.4803  0.000127  0.019721
# 80    1.4904  8e-06     0.009614
# 160   1.4953  0.0       0.004747
# 320   1.4976  0.0       0.002358
