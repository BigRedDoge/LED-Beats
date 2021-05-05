import numpy as np

def FFT(P):
    n = len(P)
    if n == 1:
        return P
    omega = np.exp((2 * np.pi * 1j) / n)
    Pe, Po = P[::2], P[1::2]
    ye, yo = FFT(Pe), FFT(Po)
    y = np.zeros(n, dtype=complex)
    for j in range(n//2):
        y[j] = ye[j] + (omega ** j) * yo[j]
        y[j + n//2] = ye[j] - (omega ** j) * yo[j]
    return y

print(FFT([5, 3, 2, 1]))