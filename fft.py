import numpy as np
from numpy.polynomial.polynomial import *
import matplotlib.pyplot as plt
import scipy.integrate as integrate


def FFT(P):
    n = len(P)
    if n == 1:
        return P
    w = np.exp((2 * np.pi * 1j) / n)
    Pe, Po = P[::2], P[1::2]
    ye, yo = FFT(Pe), FFT(Po)
    y = np.zeros(n, dtype=complex)
    for j in range(n//2):
        y[j] = ye[j] + (w ** j) * yo[j]
        y[j + n//2] = ye[j] - (w ** j) * yo[j]
    return y

def IFFT(Y):
    n = len(Y)
    k = np.linspace(-n//(2+1), n//2, n)
    w = np.exp((2 * np.pi * 1j) / n)
    seq = np.zeros(n, dtype=complex)
    for i in range(n):
        seq[i] = Y[i] * (w ** (-1*(i-1)*(k[i]-1)))
    return seq

print(FFT([2, 2]))

t = np.linspace(0, 0.5, 500)
# y = amplitude * sin(2 * pi * frequency * t)
s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)
z = np.sin(40 * 2 * np.pi * t) + np.sin(90 * 2 * np.pi * t)
z = 2 + (2 * t)
print(abs(IFFT(z)))

n = len(z)
w = np.exp((-2 * np.pi * 1j) / n)
#ifft = (1 / n) * integrate.quad(lambda x: z[x] * np.exp(-), np.ninf, np.inf)

#transformation =

#for i in range(2):
#    print("Value at index {}:\t{}".format(i, fft[i + 1]), "\nValue at index {}:\t{}".format(fft.size -1 - i, fft[-1 - i]))

plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.plot(t, z)
#plt.plot(t, s)
plt.show()