"""
This file is part of Diamond. Diamond is a confocal scanner written
in python / Qt4. It combines an intuitive gui with flexible
hardware abstraction classes.

Diamond is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Diamond is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with diamond. If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2009 Helmut Rathgen <helmut.rathgen@gmail.com>
"""

import numpy as np
import scipy.optimize, scipy.stats

########################################################
# utility functions 
########################################################

def baseline(y,n=20):
	"""
	Returns the baseline of 'y'. 'n' controls the discretization.
	The difference between the maximum and the minimum of y is discretized into 'n' steps.
	"""
	hist, bin_edges = np.histogram(y,n)
	return bin_edges[hist.argmax()]

def find_edge(y,bins=20):
	"""Returns edge of a step function"""
	h,b=np.histogram(y,bins=bins)
	# PYTHON3 int division different
	i0 = int(bins/2)
	i  = h[i0:].argmax()+i0
	threshold = 0.5*(b[0]+b[i])
	return np.where(y>threshold)[0][0]

def run_sum(y, n=10):
	"""Calculates the running sum over 'y' (1D array) in a window with 'n' samples."""

	N = len(y)
	
	yp = np.empty(N)

	for i in range(N):
		if i+n > N:
			yp[i]=yp[N-n] # pad the last array entries with the last real entry
		else:
			yp[i]=np.sum(y[i:i+n])
			
	return yp

########################################################
# non-linear least square fitting 
########################################################

def fit(x, y, model, estimator):
	"""Perform least-squares fit of two dimensional data (x,y) to model 'Model' using Levenberg-Marquardt algorithm.\n
	'Model' is a callable that takes as an argument the model parameters and returns a function representing the model.\n
	'Estimator' can either be an N-tuple containing a starting guess of the fit parameters, or a callable that returns a respective N-tuple for given x and y."""
	if callable(estimator):
		#return scipy.optimize.leastsq(lambda pp: model(*pp)(x) - y, estimator(x,y), warning=False)[0]
		p = scipy.optimize.leastsq(lambda pp: model(*pp)(x) - y, estimator(x,y))[0]
		return p
	else:
		#return scipy.optimize.leastsq(lambda pp: model(*pp)(x) - y, estimator, warning=False)[0]
		return scipy.optimize.leastsq(lambda pp: model(*pp)(x) - y, estimator)[0]

def nonlinear_model(x, y, s, model, estimator, message=False):
	"""Performs a non-linear least-squares fit of two dimensional data and a primitive error analysis. 
	
	parameters:
	
	x		  = x-data
	y		  = y-data
	s		  = standard deviation of y
	model	  = the model to use for the fit. must be a factory function
				that takes as parameters the parameters to fit and returns
				a function y(x)
	estimator = either an n-tuple (or array) containing the starting guess
				of the fit parameters or a callable that takes x and y
				as arguments and returns a starting guess

	return values:
	
	p		 = set of parameters that minimizes the chisqr
	
	cov		 = covariance matrix
	
	q		 = probability of obtaining a chisqr larger than the observed one
	
			   if 0.9 > q > 0.1 the fit is credible
			   
			   if q > 0.001, the fit may be credible if we expect that the
			   reason for the small q are non-normal distributed errors
			   
			   if q < 0.001, the fit must be questioned. Possible causes are
				   (i) the model is not suitable
				   (ii) the standard deviations s are underestimated
				   (iii) the standard deviations s are not normal distributed
				   
			   if q > 0.9, the fit must be questioned. Possible causes are
				   (i) the standard deviations are overestimated
				   (ii) the data has been manipulated to fit the model 
	
	chisqr0	 = sum over chisqr evaluated at the minimum
	"""
	chisqr = lambda p: ( model(*p)(x) - y ) / s
	if callable(estimator):
		p = estimator(x,y)
	else:
		p = estimator
	result = scipy.optimize.leastsq(chisqr, p, full_output=True)
	
	if message:
		# PYTHON3 EDIT
		print(result[4], result[3])
	p = result[0]
	cov = result[1]
	
	# there are some cases where leastsq doesn't raise an exception, however returns None for
	# the covariance matrix. To prevent 'invalid index' errors in functions that call nonlinear_model,
	# we replace the 'None' by a matrix with right dimension filled with np.NaN.
	if cov is None:
		cov = np.NaN * np.empty( (len(p),len(p)) )
	
	chi0 = result[2]['fvec']
	
	chisqr0 = np.sum(chi0**2)
	nu = len(x) - len(p)
	
	q = scipy.special.gammaincc(0.5*nu,0.5*chisqr0)
	
	return p, cov, q, chisqr0

########################################################
# standard factory function for non-linear fitting 
########################################################

def Cosinus(a, T, c):
	"""Returns a Cosinus function.
	
		f = a\cos(2\pi(x-x0)/T)+c
	
	Parameter:
	
	a	 = amplitude
	T	 = period
	x0	 = position
	c	 = offset in y-direction
	"""
	return lambda x: a*np.cos( 2*np.pi*x/float(T) ) + c

setattr(Cosinus, 'formula', r'$cos(c,a,T;x)=a\cos(2\pi x/T)+c$')

def Cosinus_phase(a, T,x0, c):
	"""Returns a Cosinus function.
	
		f = a\cos(2\pi(x-x0)/T)+c
	
	Parameter:
	
	a	 = amplitude
	T	 = period
	x0	 = position
	c	 = offset in y-direction
	"""
	return lambda x: a*np.cos( 2*np.pi*(x-x0)/float(T) ) + c

setattr(Cosinus_phase, 'formula', r'$cos(c,a,T;x)=a\cos(2\pi (x-x0)/T)+c$')


'''
def DecayCosinus(a, T, c):
	"""Returns a Cosinus function.
	
		f = a\cos(2\pi(x-x0)/T)\e\\+c
	
	Parameter:
	
	a	 = amplitude
	T	 = period
	x0	 = position
	c	 = offset in y-direction
	"""
	return lambda x: np.exp(1/) * a*np.cos( 2*np.pi*x/float(T) ) + c

setattr(Cosinus, 'formula', r'$cos(c,a,T;x)=a\cos(2\pi x/T)+c$')
'''

def CosinusEstimator(x, y):
	c = y.mean()
	a = 2**0.5 * np.sqrt( ((y-c)**2).sum() )
	# better to do estimation of period from
	Y = np.fft.fft(y)
	N = len(Y)
	D = float(x[1] - x[0])
	i = abs(Y[1:int(N/2)+1]).argmax()+1
	T = (N * D) / i
	return a, T, c

def CosinusNoOffset(a, T):
	"""Returns a Cosinus function without constant offset.
	
		f = a\cos(2\pi(x-x0)/T)
	
	Parameter:
	
	a	 = amplitude
	T	 = period
	x0	 = position
	"""
	return lambda x: a*np.cos( 2*np.pi*x/float(T) )

setattr(CosinusNoOffset, 'formula', r'$cos(a,T;x)=a\cos(2\pi x/T)$')

def CosinusNoOffsetEstimator(x, y):
	a = 2**0.5 * np.sqrt( (y**2).sum() )
	# better to do estimation of period from
	Y = np.fft.fft(y)
	N = len(Y)
	D = float(x[1] - x[0])
	i = abs(Y[1:int(N/2)+1]).argmax()+1
	T = (N * D) / i
	return a, T

def ExponentialZero(a, w, c):
	"""Exponential centered at zero.
	
		f = a*exp(-x/w) + c
	
	Parameter:
	
	a	 = amplitude
	w	 = width
	c	 = offset in y-direction
	"""
	return lambda x: a*np.exp(-x/w)+c

def ExponentialZeroEstimator(x, y): 
	"""Exponential Estimator without offset. a*exp(-x/w) + c"""
	c=y[-1]
	a=y[0]-c
	w=x[-1]*0.5
	return a, w, c

def GaussianZero(a, w, c):
	"""Gaussian function centered at zero.
	
		f = a*exp(-(x/w)**2) + c
	
	Parameter:
	
	a	 = amplitude
	w	 = width
	c	 = offset in y-direction
	"""
	return lambda x: a*np.exp( -(x/w)**2 ) + c

setattr(GaussianZero, 'formula', r'$f(a,w,c;x)=a\exp(-(x/w)^2)+c$')

def GaussianZeroEstimator(x, y): 
	"""Estimator for GaussianZero: a*exp(-0.5*(x/w)**2) + c"""
	c=y[-1]
	a=y[0]-c
	w=x[-1]*0.5
	return a, w, c

def Gaussian(c, a, x0, w):
	"""Gaussian function.
	
		f = a*exp( -0.5(x-x0)**2 / w**2 ) + c
	
	Parameter:
	
	a	 = amplitude
	w	 = width
	c	 = offset in y-direction
	"""
	return lambda x: c + a*np.exp( -0.5*((x-x0)/w)**2	)

setattr(Gaussian, 'formula', r'$f(c,a,x0,w;x)=c+a\exp(-0.5(x-x0)^2/w^2)$')

def ExponentialPowerZero(a, w, p, c):
	"""Exponential decay with variable power centered at zero.
	
		f = a*exp(-(x/w)**p) + c
	
	Parameter:
	
	a	 = amplitude
	w	 = width
	p	 = power
	c	 = offset in y-direction
	"""
	return lambda x: a*np.exp( -(x/w)**p ) + c

setattr(ExponentialPowerZero, 'formula', r'$f(a,w,p,c;x)=a\exp(-(x/w)^p)+c$')

def ExponentialPowerZeroEstimator(x, y): 
	"""Estimator for exponential decay with variable offset."""
	c=y[-1]
	a=y[0]-c
	w=x[-1]*0.5
	return a, w, 2, c


def GaussianZeroEstimator(x, y): 
	"""Gaussian Estimator without x offset. c+ a*exp( -0.5*(x/w)**2)"""
	a=y.argmax()
	#x0=x[y.argmax()]
	w=x[(len(x)/2)]
	c=(min(y)+max(y))/2
	return a, w, c


def DoubleGaussian(a1, a2, x01, x02, w1, w2):
	"""Gaussian function with offset."""
	return lambda x: a1*np.exp( -0.5*((x-x01)/w1)**2   ) + a2*np.exp( -0.5*((x-x02)/w2)**2	 )

setattr(DoubleGaussian, 'formula', r'$f(c,a1, a2,x01, x02,w1,w2;x)=a_1\exp(-0.5((x-x_{01})/w_1)^2)+a_2\exp(-0.5((x-x_{02})/w_2)^2)$')

def DoubleGaussianEstimator(x, y):
	center = (x*y).sum() / y.sum()
	ylow = y[x < center]
	yhigh = y[x > center]
	x01 = x[ylow.argmax()]
	x02 = x[len(ylow)+yhigh.argmax()]
	a1 = ylow.max()
	a2 = yhigh.max()
	w1 = w2 = center**0.5
	return a1, a2, x01, x02, w1, w2

# important note: lorentzian can also be parametrized with an a' instead of a,
# such that a' is directly related to the amplitude (a'=f(x=x0)). In this case a'=a/(pi*g)
# and f = a * g**2 / ( (x-x0)**2 + g**2 ) + c.
# However, this results in much poorer fitting success. Probably the g**2 in the numerator
# causes problems in Levenberg-Marquardt algorithm when derivatives
# w.r.t the parameters are evaluated. Therefore it is strongly recommended
# to stick to the parametrization given below.
def Lorentzian(x0, g, a, c):
	"""Lorentzian centered at x0, with amplitude a, offset y0 and HWHM g."""
	return lambda x: a / np.pi * (	g / ( (x-x0)**2 + g**2 )  ) + c

setattr(Lorentzian, 'formula', r'$f(x0,g,a,c;x)=a/\pi (g/((x-x_0)^2+g^2)) + c$')

def LorentzianEstimator(x, y):
	c = scipy.stats.mode(y)[0][0]
	yp = y - c
	Y = np.sum(yp) * (x[-1] - x[0]) / len(x)
	ymin = yp.min()
	ymax = yp.max()
	if ymax > abs(ymin):
		y0 = ymax
	else:
		y0 = ymin
	x0 = x[y.argmin()]
	g = Y / (np.pi * y0)
	a = y0 * np.pi * g
	return x0, g, a, c

def LorentzianWithoutOffset(x0, g, a):
	"""Lorentzian centered at x0, with amplitude a, and HWHM g."""
	return lambda x: a / np.pi * (	g / ( (x-x0)**2 + g**2 )  )

def Antibunching(alpha, c, tau, t0):
	"""Antibunching. g(2) accounting for Poissonian background."""
	return lambda t: c*(1-alpha*np.exp(-(t-t0)/tau))

setattr(Antibunching, 'formula', r'$g(\alpha,c,\tau,t_0;t)=c(1 - \alpha \exp(-(t-t_0)/\tau))$')

def FCSTranslationRotation(alpha, tau_r, tau_t, N):
	"""Fluorescence Correlation Spectroscopy. g(2) accounting for translational and rotational diffusion."""
	return lambda t: (1 + alpha*np.exp(-t/tau_r) ) / (N * (1 + t/tau_t) )

setattr(FCSTranslationRotation, 'formula', r'$g(\alpha,\tau_R,\tau_T,N;t)=\frac{1 + \alpha \exp(-t/\tau_R)}{N (1 + t/\tau_T)}$')

def FCSTranslation(tau, N):
	"""Fluorescence Correlation Spectroscopy. g(2) accounting for translational diffusion."""
	return lambda t: 1. / (N * (1 + t/tau) )

setattr(FCSTranslation, 'formula', r'$g(\tau,N;t)=\frac{1}{N (1 + t/\tau)}$')

def SumOverFunctions( functions ):
	"""Creates a factory that returns a function representing the sum over 'functions'.
	'functions' is a list of functions. 
	The resulting factory takes as arguments the parameters to all functions,
	flattened and in the same order as in 'functions'."""
	def function_factory(*args):
		def f(x):
			y = np.zeros(x.shape)
			i = 0
			for func in functions:
				n = func.func_code.co_argcount
				y += func(*args[i,i+n])(x)
				i += n
		return f
	return function_factory

def NLorentzians(*p):

	# PYTHON3 UPDATE true division
	N = int((len(p)-1)/3)
	def f(x):
		y = p[0]*np.ones(x.shape)
		i = 0
		for i in range(N):
			y += LorentzianWithoutOffset(*p[i*3+1:i*3+4])(x)
		return y   
	return f

def brot_transitions_upper(B, D, E, phase):
	return lambda theta: 3./2. * B**2/D * np.sin(theta + phase)**2 + ( B**2 * np.cos(theta + phase)**2 + (E + B**2/(2*D) * np.sin(theta+phase)**2)**2)**0.5 + D
	
def brot_transitions_lower(B, D, E, phase):
	return lambda theta: 3./2. * B**2/D * np.sin(theta + phase)**2 - ( B**2 * np.cos(theta + phase)**2 + (E + B**2/(2*D) * np.sin(theta+phase)**2)**2)**0.5 + D


#################################################################
# convenience functions for performing some frequently used fits
#################################################################

def grow(mask):
	"""Grows regions in a 1D binary array in both directions.
	Helper function for multiple Lorentzian fit."""
	return np.logical_or(np.logical_or(mask, np.append(mask[1:],False)), np.append(False,mask[:-1]))

def fit_multiple_lorentzians(x,y,number_of_lorentzians='auto',threshold=0.5):

	"""Attempts to fit a sum of multiple Lorentzians and returns the fit parameters (c, x0, g0, a0, x1, g1, a1,... )."""
	# first re-scale the data to the range (0,1), such that the baseline is at 0.
	# flip the data in y-direction if threshold is negative 
	y0 = baseline(y)
	yp = y - y0
	if threshold < 0:
		yp = -yp
	y_max = yp.max()
	yp = yp / y_max
	# compute crossings through a horizontal line at height 'threshold'
	mask = yp>abs(threshold)
	edges = np.where(np.logical_xor(mask, np.append(False,mask[:-1])))[0]
	if len(edges)%2 != 0:
		raise RuntimeError('uneven number of edges')
	if len(edges) < 2:
		raise RuntimeError('no peak to fit')
	if number_of_lorentzians is 'auto': # try to find N automatically
		# attempt initial growth of connected regions to kill noise 
		while True:
			mask = grow(mask)
			new_edges = np.where(np.logical_xor(mask, np.append(False,mask[:-1])))[0]
			if len(new_edges) < len(edges):
				edges = new_edges
			else:
				break
		if len(edges)%2 != 0:
			raise RuntimeError('uneven number of edges')
		# if there is more than one region, throw away small regions and
		# keep only those regions that are larger than half of the largest region
		# otherwise use all regions

		# PYTHON3 UPDATE true division
		N = int(len(edges)/2)
		left_and_right_edges = edges.reshape((N,2))

		#if len(edges)/2 > 1:
		#	 widths = left_and_right_edges[:,1] - left_and_right_edges[:,0]
		#	 left_and_right_edges = left_and_right_edges[ np.where(widths>0.5*widths.max())[0], : ] 
		#	 N = left_and_right_edges.shape[0]
	else: # if N is specified grow until number of regions =< N
		while len(edges)/2 > number_of_lorentzians:
			mask = grow(mask)
			edges = np.where(np.logical_xor(mask, np.append(False,mask[:-1])))[0]
		if len(edges)%2 != 0:
			raise RuntimeError('uneven number of edges')
		
		# PYTHON3 UPDATE true division
		N = int(len(edges)/2)
		left_and_right_edges = edges.reshape((N,2))
	p = [ 0 ]
	# for every local maximum, estimate parameters of Lorentzian and append them to the list of parameters p
	for left, right in left_and_right_edges:
		g = abs(x[right] - x[left]) # FWHM
		i = y[left:right].argmax()+left # index of local minimum
		x0 = x[i] # position of local minimum
		a = y[i] * np.pi * g # height of local minimum in terms of Lorentzian parameter a
		p += [ x0, g, a ]

	p = tuple(p)

	# chi for N Lorentzians with a common baseline
	def chi(p):
		ypp = p[0]-yp
		for i in range(N):
			ypp += LorentzianWithoutOffset(*p[i*3+1:i*3+4])(x)
		return ypp

	r = scipy.optimize.leastsq(chi, p, full_output=True)

	if r[-1] == 0:
		raise RuntimeError('least square fit did not work out')	   

	p = np.array(r[0])

	# rescale fit parameters back to original data 
	p[0] = p[0]*y_max*np.sign(threshold) + y0
	p[3::3] *= y_max*np.sign(threshold)

	return p

def find_local_maxima(y,n):
	"Returns the indices of the n largest local maxima of y."

	half = 0.5*y.max()
	mask = y>half
	
	# get left and right edges of connected regions
	
	right_shifted = np.append(False, mask[:-1])
	left_shifted = np.append(mask[1:], False)
	
	left_edges =  np.where( np.logical_and(mask,np.logical_not(right_shifted) ))[0]
	right_edges = np.where( np.logical_and(mask,np.logical_not(left_shifted)) )[0] + 1

	if len(left_edges) < n:
		raise RuntimeError('did not find enough edges')
	
	indices = []
	for k in range(len(left_edges)):
		left = left_edges[k]
		right = right_edges[k]
		indices.append( y[left:right].argmax()+left )
	indices = np.array(indices)
	maxima = y[indices]
	indices = indices[maxima.argsort()][::-1]
	return indices[:n]

def find_n14hf_maxima(y,n,dx):
	"Returns the indices of the n largest local maxima of y."

	hf=2.18e6/dx
	half = 0.5*y.max()
	mask = y>half
	
	# get left and right edges of connected regions
	
	right_shifted = np.append(False, mask[:-1])
	left_shifted = np.append(mask[1:], False)
	
		
	left_edges =  np.where( np.logical_and(mask,np.logical_not(right_shifted) ))[0]
	right_edges = np.where( np.logical_and(mask,np.logical_not(left_shifted)) )[0] + 1

	if len(left_edges) < n:
		raise RuntimeError('did not find enough edges')
	
	indices = []
	for k in range(len(left_edges)):
		left = left_edges[k]
		right = right_edges[k]
		indices.append( y[left:right].argmax()+left )
	indices = np.array(indices)
	maxima = y[indices]
	indices = indices[maxima.argsort()][::-1]
	return indices[:n]


"""
def fit_rabi(x, y, s):
	y_offset=y.mean()
	yp = y - y_offset

	p = fit(x, yp, CosinusNoOffset, CosinusNoOffsetEstimator)
	if p[0] < 0:
		p[0] = -p[0]
		p[2] =	( ( p[2]/p[1] + 0.5 ) % 1 ) * p[1]
		p = fit(x, yp, CosinusNoOffset, p)
	p = (p[0], p[1], p[2], y_offset)
	result = nonlinear_model(x, y, s, Cosinus, p)
	p = result[0]
	if p[2]>0.5*p[1]:
		while(p[2]>0.5*p[1]):
			p[2] -= p[1]
		result = nonlinear_model(x, y, s, Cosinus, p)
	return result
"""

def fit_rabi(x, y, s):
	y_offset=y.mean()
	yp = y - y_offset

	p = fit(x, yp, CosinusNoOffset, CosinusNoOffsetEstimator)
	if p[0] < 0:
		p[0] = -p[0]
		p[2] =	( ( p[2]/p[1] + 0.5 ) % 1 ) * p[1]
		#p = fit(x, yp, CosinusNoOffset, p)
	p = (p[0], p[1], y_offset)
	return nonlinear_model(x, y, s, Cosinus, p)

def fit_decay_rabi(x, y, s):
	y_offset=y.mean()
	yp = y - y_offset

	p = fit(x, yp, CosinusNoOffset, CosinusNoOffsetEstimator)
	if p[0] < 0:
		p[0] = -p[0]
		p[2] =	( ( p[2]/p[1] + 0.5 ) % 1 ) * p[1]
		#p = fit(x, yp, CosinusNoOffset, p)
	p = (p[0], p[1], y_offset)
	return nonlinear_model(x, y, s, Cosinus, p)

def extract_pulses(y):
	"""
	Extracts pi, pi/2 and 3pi/2 pulses from a Rabi measurement.
	
	Parameters:
		y = the arry containing y data
		
	Returns:
		f, r, pi, 2pi = arrays containing the indices of the respective pulses and their multiples 
	"""
	# The goal is to find local the rising and falling edges and local minima and maxima.
	# First we estimate the 'middle line' by the absolute minimum and maximum.
	# Then we cut the data into sections below and above the middle line.
	# For every section we compute the minimum, respectively maximum.
	# The falling and rising edges mark multiples of pi/2, respectively 3pi/2 pulses.

	# center line
	m=0.5*(y.max()+y.min())
	
	# boolean array containing positive and negative sections
	b = y < m
	
	# indices of rising and falling edges
	# rising edges: last point below center line
	# falling edges: last point above center line
	rising = np.where(b[:-1]&~b[1:])[0]
	falling = np.where(b[1:]&~b[:-1])[0]

	# local minima and maxima
	
	pi = [ y[:rising[0]].argmin() ]
	two_pi = [ y[:falling[0]].argmax() ]
	
	for i in range(1,len(rising)):
		pi.append( rising[i-1] + y[rising[i-1]:rising[i]].argmin() )
		
	for i in range(1,len(falling)):
		two_pi.append(falling[i-1] + y[falling[i-1]:falling[i]].argmax() )

	# For rising edged, we always use the last point below the center line,
	# however due to finite sampling and shot noise, sometimes
	# the first point above the line may be closer to the actual zero crossing
	for i, edge in enumerate(rising):
		if y[edge+1]-m < m-y[edge]:
			rising[i] += 1
	# similarly for the falling edges
	for i, edge in enumerate(falling):
		if m-y[edge+1] < y[edge]-m:
			falling[i] += 1
	
	return np.array(falling), np.array(rising), np.array(pi), np.array(two_pi)
#################################################################
# perform 2d normalized gaussian fit
#################################################################
def multivariate_gaussian(pos, mu, Sigma):
	"""Return the multivariate Gaussian distribution on array pos.

	pos is an array constructed by packing the meshed arrays of variables
	x_1, x_2, x_3, ..., x_k into its _last_ dimension.

	"""

	Sigma_inv = np.linalg.inv(Sigma)
	N = 1
	# This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
	# way across all the input variables.
	fac = np.einsum('...k,...kl,...l->...', pos-mu, Sigma_inv, pos-mu)

	return np.exp(-fac / 2) / N

def multivariate_gaussian_model(pos):
	return lambda p:multivariate_gaussian(pos, np.array([p[0],p[1]]), np.array([[p[2],0],[0,p[3]]]))

def twodim_gaussian_fit(data):
	"""
	fitting the normalized 2d data
	Initial guess has a shape of (6,), the first two elements are the mean vector 
	the rests are elements of the cov matrix 
	"""

	xx, yy = np.meshgrid(np.linspace(0, data.shape[1]-1, len(data[0])), np.linspace(0, data.shape[0]-1, len(data)))
	data_pos = np.empty(xx.shape+(2,))
	data_pos[:,:,0] = xx
	data_pos[:,:,1] = yy

	def cost_func(p):
		cost = (multivariate_gaussian_model(data_pos)(p)-data).flatten()
		return cost

	initial_guess = [(data.shape[1]-1)/2, (data.shape[0]-1)/2, data.shape[1]/2, data.shape[0]/2]
	r = scipy.optimize.leastsq(cost_func, initial_guess,
					full_output=True
					)
					# bounds=([0, 0, 0, 0],
							# [data.shape[1]-1, data.shape[0]-1, 4*data.shape[1], 4*data.shape[0]])
							# )
	return r[0], data_pos
#scipy.optimize.leastsq(chi, p, full_output=True)

if __name__ == '__main__':
	
	import cPickle
	d = cPickle.load(open('point14_ESR_102000cts_cwODMR04_ODMR.pys','rb'))
	#d = cPickle.load(open('2012-01-25_tsukubac12_nv39_pulsed_precise_R_ODMR.pys','rb'))
	
	x = d['frequency']
	y = d['counts']
	
	number_of_lorentzians='auto'
	
	threshold=0.5

	y0 = baseline(y)
	yp = y - y0
	if threshold < 0:
		yp = -yp
	y_max = yp.max()
	yp = yp / y_max
	mask = yp>abs(threshold)
	# get left and right edges of connected regions	   
	edges = np.where(np.logical_xor(mask, np.append(False,mask[:-1])))[0]
	if len(edges)%2 != 0:
		raise RuntimeError('uneven number of edges')
	if len(edges) < 2:
		raise RuntimeError('no peak to fit')
	if number_of_lorentzians is 'auto': # try to find N automatically
		# attempt initial growth to kill noise 
		while True:
			mask = grow(mask)
			new_edges = np.where(np.logical_xor(mask, np.append(False,mask[:-1])))[0]
			if len(new_edges) < len(edges):
				edges = new_edges
			else:
				break
		if len(edges)%2 != 0:
			raise RuntimeError('uneven number of edges')
		# if there is more than one region, throw away small regions and
		# keep only those regions that are larger than half of the largest region
		# otherwise use all regions
		N = len(edges)/2
		left_and_right_edges = edges.reshape((N,2))
		#if len(edges)/2 > 1:
		#	 widths = left_and_right_edges[:,1] - left_and_right_edges[:,0]
		#	 left_and_right_edges = left_and_right_edges[ np.where(widths>0.5*widths.max())[0], : ] 
		#	 N = left_and_right_edges.shape[0]
	else: # if N is specified grow until number of regions =< N
		while len(edges)/2 > number_of_lorentzians:
			mask = grow(mask)
			edges = np.where(np.logical_xor(mask, np.append(False,mask[:-1])))[0]
		if len(edges)%2 != 0:
			raise RuntimeError('uneven number of edges')
		N = len(edges)/2
		left_and_right_edges = edges.reshape((N,2))
	p = [ 0 ]
	# for every local maximum, estimate parameters of Lorentzian and append them to the list of parameters p
	for left, right in left_and_right_edges:
		g = abs(x[right] - x[left]) # FWHM
		i = y[left:right].argmax()+left # index of local minimum
		x0 = x[i] # position of local minimum
		a = y[i] * np.pi * g # height of local minimum in terms of Lorentzian parameter a
		p += [ x0, g, a ]

	p = tuple(p)

	# chi for N Lorentzians with a common baseline
	def chi(p):
		ypp = p[0]-yp
		for i in range(N):
			ypp += LorentzianWithoutOffset(*p[i*3+1:i*3+4])(x)
		return ypp

	r = scipy.optimize.leastsq(chi, p, full_output=True)

	if r[-1] == 0:
		raise RuntimeError('least square fit did not work out')	   

	p = np.array(r[0])

	p[0] = p[0]*y_max*np.sign(threshold) + baseline
	p[3::3] *= y_max*np.sign(threshold)

	yy=NLorentzians(*p)(x)
	
	
	