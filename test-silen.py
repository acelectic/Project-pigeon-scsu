from scipy.io.wavfile import write
from numpy import arange, pi, sin, int16
from playsound import playsound

def f(t, f_c, f_m, beta):
    # t    = time
    # f_c  = carrier frequency
    # f_m  = modulation frequency
    # beta = modulation index
    return sin(2*pi*f_c*t - beta*sin(2*f_m*pi*t))

def to_integer(signal):
    # Take samples in [-1, 1] and scale to 16-bit integers,
    # values between -2^15 and 2^15 - 1.
    return int16(signal*(2**15 - 1))

N = 48000 # samples per second
x = arange(4*N) # three seconds of audio

data = f(x/N, 1500, 2, 100)
write("slow.wav", N, to_integer(data))

data = f(x/N, 1500, 8, 100)
write("fast.wav", N, to_integer(data))


file_silen = "fast.wav"
playsound(file_silen)