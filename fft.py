import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import sounddevice as sd
import time
import math


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
    N = len(Y)
    coeffs = np.zeros(N, dtype=complex)
    for n in range(N):
        inverse = 0
        for m in range(len(coeffs)):
            inverse += Y[m] * np.exp((2 * 1j * np.pi * m * n) / N)
        coeffs[n] = inverse * (1 / N)
    return coeffs

def IFFT2(Y):
    conj_before = [np.conjugate(yo) for yo in Y]
    conj_after = [np.conjugate(f) / len(Y) for f in FFT(conj_before)] 
    return conj_after


def FFT_freq(n, d=1):
    val = 1.0 / (n * d)
    results = np.empty(n, dtype=int)
    N = (n-1)//2 + 1
    p1 = np.arange(0, N, dtype=int)
    results[:N]
    p2 = np.arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    return results * val


t = np.linspace(0, 0.5, 500)
f = np.linspace(0, 100, 500)
# y = amplitude * sin(2 * pi * frequency * t)
s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)
z = np.sin(40 * 2 * np.pi * t) + np.sin(90 * 2 * np.pi * t)
#fft = np.fft.fft(z)
#print(IFFT2([40, 90]))

fs_rate, signal = wavfile.read("hotandcold.wav")
#fs_rate, signal = 1000, z
#signal = signal[:len(signal)//32]
"""
if len(signal.shape) == 2:
    signal = signal.sum(axis=1) / 2
N = signal.shape[0]
secs = N / float(fs_rate)
Ts = 1.0 / fs_rate
t = np.arange(0, secs, Ts)
#fft = abs(FFT(signal))
fft = abs(np.fft.fft(signal))
fft_side = fft[range(N//2)]
#freqs = FFT_freq(signal.size, t[1]-t[0])
freqs = np.fft.fftfreq(signal.size, t[1]-t[0])
freqs_side = freqs[range(N//2)]
p1 = plt.plot(t, signal, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

p2 = plt.plot(freqs, fft, "r") # plotting the complete fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')

p3 = plt.plot(freqs_side, abs(fft_side), "b") # plotting the positive fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')
plt.show()

coef_freq = list(zip(fft_side, freqs_side))
coef_freq.sort(reverse=True, key=lambda x: x[0])
print(coef_freq[:5])
#    if coef:
#        print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef,
#                                                    f=freq))
reconstruct = abs(np.fft.ifft(np.fft.fft(signal)))
#reconstruct = [abs(x) for x in IFFT2(np.fft.fft(signal))]
print("IFFT calculated")
sd.play(reconstruct, fs_rate)
time.sleep(secs)
sd.stop()
"""

print(len(signal))

def process_audio(signal, fs_rate, chunk_size=10025):
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0] / chunk_size
    secs = N / float(fs_rate)
    Ts = 1.0 / fs_rate
    t = np.linspace(0, secs, chunk_size//2)
    #t = np.arange(0, secs, Ts)
    sqre = math.ceil(np.sqrt(N))
    plots = []

    for i in range(0, chunk_size*(len(signal)//chunk_size) , chunk_size):
        chunk = signal[i:i+chunk_size] 
        chunk_side = chunk[range(len(chunk)//2)]
        fft = abs(np.fft.fft(chunk))
        fft_side = fft[range(fft.shape[0]//2)]

        if max(fft) != 0:
            freqs = np.fft.fftfreq(chunk.size, t[1]-t[0])
            freqs_side = freqs[range(fft.shape[0]//2)]
            plots.append(zip(chunk, freqs_side, fft_side))

    fig, ax = plt.subplots(5, 2)
    for i in range(5):
        p = list(zip(*plots[np.random.randint(0, len(plots))]))
        print(np.asarray(p).shape)
        chunk, freqs, fft = p[0], p[1], p[2]
        
        ax[i][0].plot(t, chunk, "g") # plotting the signqal
        ax[i][0].set(xlabel='Time (s)', ylabel='Amplitude')

        ax[i][1].plot(freqs, fft, "r") # plotting the complete fft spectrum
        ax[i][1].set(xlabel='Frequency (Hz)', ylabel='Count double-sided')
        ax[i][1].set_xlim([-10, 10000])

    plt.show()


process_audio(signal, fs_rate)

#p1 = plt.plot(t, IFFT(fft), "g") # plotting the signal
#plt.xlabel('Time')
#plt.ylabel('Amplitude')
#plt.show()

#transformation =

#for i in range(2):
#    print("Value at index {}:\t{}".format(i, fft[i + 1]), "\nValue at index {}:\t{}".format(fft.size -1 - i, fft[-1 - i]))

#plt.ylabel("Amplitude")
#plt.xlabel("Frequency (s)")
#plt.plot(f, fft)
#plt.plot(t, s)
#plt.show()