import numpy as np
from scipy.fftpack import fft,ifft

def myfft(x, window, nfft):
	xn = np.zeros(nfft)
	for tmpIndex in range(x.size):
		xn[tmpIndex] = x[tmpIndex]*window[tmpIndex]
	X = fft(xn,nfft)					#frequency of x
	# absX = abs(X)				
	# mX = X[range(nfft/2+1)]
	# mag = absX
	# mag = absX*2
	# mag[0] = mag[0]*2
	# mag[nfft/2] = mag[nfft/2]*2		#mag of X
	return X

def getPolorMag(X ,nfft):
	mX = X[range(nfft/2+1)]
	return mX

def magConverse(mX, length_of_signal):
	mag = mX/length_of_signal*2
	mag[0] = mag[0]/2
	lastIndex = len(mX) - 1
	mag[lastIndex] = mag[lastIndex]/2
	return mag

def getWindowCoefficient(windowName):
	if windowName == 'hanning':
		return 2
	elif windowName == 'hamming':
		return 1.852
	else:
		return 0

def myifft(mX, window):
	xn = np.zeros(len(window))
	tmpx = ifft(mX)
	for tmpIndex in range(len(window)):
		xn[tmpIndex] = np.real(tmpx[tmpIndex])
		xn[tmpIndex] = xn[tmpIndex]/window[tmpIndex]
	return xn