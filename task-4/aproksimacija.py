import matplotlib.pyplot as plt
import numpy as np
import math

lines = list(map(lambda l: l.split('\t'), open('./duom5.txt').read().split('\n')[:-1]))
x_axis = list(map(lambda l: float(l[0]), lines))
y_axis = list(map(lambda l: float(l[1]), lines))

plt.plot(x_axis, y_axis, 'ro')

m = 4
n = len(x_axis)
period = 3 #maybe change to pi
lam = 2*math.pi/period

def fi(j, x, alpha):
    return math.cos(alpha*j*x) if j % 2 == 0 else math.sin(alpha*j*x)
    # return math.cos(alpha*(j/2)*x) if j % 2 == 0 else math.sin(alpha*((j+1)/2)*x)

B = np.zeros((n, m + 1))
print('m', m)
print('n', n)
for i in range(0, n):
    for j in range(0, m + 1):
        B[i][j] = fi(j, x_axis[i], period)


BT = np.transpose(B)
res = np.dot(BT, y_axis)
prod = np.dot(BT, B)
# print('B', B)
# print('BT', BT)
# print('RES', res)
# print('PROD', prod)
bbb = np.linalg.solve(prod, res)
print('bbb', bbb)
def f(a, x):
 return a[0] + a[1]*math.cos(lam*x) + a[2]*math.sin(lam*x) + a[3]*math.cos(2*lam*x) + a[4]*math.sin(2*lam*x)

fx = []
fy = []
for i in np.arange(0, 10, 0.1):
    fy.append(f(bbb, i))
    fx.append(i)


plt.plot(fx, fy)

def moar(a, x):
    sum = 0
    for j in range(0, len(a)):
        sum += a[j] * fi(j, x, period)
    return sum

mx = []
my = []
for i in np.arange(0, 10, 0.1):
    my.append(f(bbb, i))
    mx.append(i)
plt.plot(mx, my)


error = 0
for i in range(0, len(x_axis)):
    error += (f(bbb, x_axis[i]) - y_axis[i])**2
error = math.sqrt(error / len(x_axis))

print('error', error)
plt.show()