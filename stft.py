#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:37:54 2018

@author: al
"""

import scipy, pylab

def stft(x, fs, framesz,hop,):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[:,:,i:i+framesamp]) 
                     for i in range(0, x.shape[-1]-framesamp, hopsamp)])
    X = X.transpose(1,2,0,3)
    return X

def istft(X, fs, T, hop):
    x = scipy.zeros((X.shape[0],X.shape[1], int(T*fs)))
    framesamp = X.shape[-1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, x.shape[-1]-framesamp, hopsamp)):
        x[:,:,i:i+framesamp] += scipy.real(scipy.ifft(X[:,:,n]))
    return x

if __name__ == '__main__':
    f0 = 440         # Compute the STFT of a 440 Hz sinusoid
    fs = 250        # sampled at 8 kHz
    T = 4.5            # lasting 5 seconds
    framesz = 0.050  # with a frame size of 50 milliseconds
    hop = 0.025      # and hop size of 25 milliseconds.

    # Create test signal and STFT.
    t = scipy.linspace(0, T, int(T*fs), endpoint=False)
#    x = scipy.sin(2*scipy.pi*f0*t)
    
    X = stft(x, fs, framesz, hop)

    # Plot the magnitude spectrogram.
    pylab.figure()
    pylab.imshow(scipy.absolute(X.T), origin='lower', aspect='auto',
                 interpolation='nearest')
    pylab.xlabel('Time')
    pylab.ylabel('Frequency')
    pylab.show()

    # Compute the ISTFT.
    xhat = istft(X, fs, T, hop)

    # Plot the input and output signals over 0.1 seconds.
    T1 = int(0.45*fs)

    pylab.figure()
    pylab.plot(t[:T1], x[:T1],'r', t[:T1], xhat[:T1],'b',)
    pylab.xlabel('Time (seconds)')

    pylab.figure()
    pylab.plot(t[-T1:], x[-T1:], t[-T1:], xhat[-T1:])
    pylab.xlabel('Time (seconds)')