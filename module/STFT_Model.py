import numpy as np
import scipy
from array import *
from FFT_Model import myfft,myifft,getPolorMag
import matplotlib.pyplot as plt

def mystft(x, window, nfft, hopsize):
	"""
	"""
	size_of_signal = len(x)
	size_of_window = len(window)
	size_of_fft = nfft
	size_of_bins = size_of_signal%hopsize

	arr_X = np.array([])
	arr_mX = np.array([])

	index_of_current_sample = 0
	while index_of_current_sample <= size_of_signal-size_of_window:
		X = myfft(x[index_of_current_sample:index_of_current_sample+size_of_window-1],window,nfft)		
		mX = getPolorMag(X,nfft)
		if index_of_current_sample == 0:
			arr_X = np.array([X])
			arr_mX = np.array([mX])
		else:
			arr_X = np.vstack((arr_X, np.array([X])))
			arr_mX = np.vstack((arr_mX, np.array([mX])))
		index_of_current_sample += hopsize
		

	return arr_X,arr_mX

def myistft(X, window, size_of_signal, hopsize):
	x = np.zeros(size_of_signal)
	size_of_frames = X[:,0].size
	for i in range(size_of_frames):
		x1 = myifft(X[i,:],window)
		x[i*hopsize:i*hopsize+len(x1)] = x1[:]
	return x