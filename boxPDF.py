# Compute the probability density of pair separations for a box geometry using the equations derived 
# 		in J. Philip. 2007 See: https://people.kth.se/~johanph/habc.pdf

# This code is reasonably fast so it isn't parallelized, but you can still call it with multiple ranks and only the root will do the computation

from __main__ import *

# From __main__:
# size (int) the number of processors
# rank (int) the processor label (starts from 0)
# helper.py

mp.dps = 25 # Only used if highRes is set to True or the errorThresh is not met

#######################################################################################################################
#  Equations 17 - 21 - rectangular prism
#######################################################################################################################
def h11(u, a, b, c, highRes):
	prefactor = 1.0 / (3*a**2*b**2*c**2)

	if highRes:
		if 0 < u <= b**2:
			factor = - 3*np.pi*b*c*u + 4*b*u**(3/2.0)
		elif b**2 < u <= c**2:
			factor = 4*b**4 + 6*b**2*c*(u - b**2)**0.5 - 6*b*c*u*mp.asin(b/u**0.5)
		elif c**2 < u <= b**2 + c**2:
			factor = 4*b**4 + 6*b**2*c*(u - b**2)**0.5 + 6*b*c*u*(mp.acos(c/u**0.5) - mp.asin(b/u**0.5)) \
						 - 2*b*(2*u + c**2)*(u - c**2)**0.5
		else:
			factor = 0
	else:
		factor = np.zeros(len(u), dtype=np.float64)
		i = np.where((0 < u) & (u <= b**2))
		factor[i] = - 3*np.pi*b*c*u[i] + 4*b*u[i]**(3/2.0)
		j = np.where((b**2 < u) & (u <= c**2))
		factor[j] = 4*b**4 + 6*b**2*c*np.sqrt(u[j] - b**2) - 6*b*c*u[j]*np.arcsin(b/np.sqrt(u[j]))
		k = np.where((c**2 < u) & (u <= b**2 + c**2))
		factor[k] = 4*b**4 + 6*b**2*c*np.sqrt(u[k] - b**2) + 6*b*c*u[k]*(np.arccos(c / np.sqrt(u[k])) - np.arcsin(b / np.sqrt(u[k]))) \
						 - 2*b*(2*u[k] + c**2)*np.sqrt(u[k] - c**2)

	return prefactor*factor

def h12(u, a, b, c, highRes):
	prefactor = 1.0 / (6*a**2*b**2*c**2)

	if highRes:
		if 0 < u <= a**2:
			factor = 12*np.pi*a*b*c*u**0.5 - 6*np.pi*a*(b + c)*u + 8*(a + c)*u**(3/2.0) - 3*u**2
		elif a**2 < u <= c**2:
			factor = 5*a**4 - 6*np.pi*a**3*b + 12*np.pi*a*b*c*u**0.5 + 8*c*u**(3/2.0) - 12*np.pi*a*b*c*(u - a**2)**0.5 \
						- 8*c*(u - a**2)**(3/2.0) - 12*a*c*u*mp.asin(a / u**0.5)
		elif c**2 < u <= a**2 + c**2:
			factor = 5*a**4 - 6*np.pi*a**3*b + 6*np.pi*a*b*c**2 - c**4 + 6*(np.pi*a*b + c**2)*u + 3*u**2 \
						- 12*np.pi*a*b*c*(u - a**2)**0.5 - 8*c*(u - a**2)**(3/2) - 4*a*(2*u + c**2)*(u - c**2)**0.5 \
						+ 12*a*c*u*(mp.acos(c/u**0.5) - mp.asin(a/u**0.5))
		else:
			factor = 0
	else:
		prefactor = 1.0 / (6*a**2*b**2*c**2)
		factor = np.zeros(len(u), dtype=np.float64)
		i = np.where((0 < u) & (u <= a**2))
		factor[i] = 12*np.pi*a*b*c*np.sqrt(u[i]) - 6*np.pi*a*(b + c)*u[i] + 8*(a + c)*u[i]**(3/2.0) - 3*u[i]**2
		j = np.where((a**2 < u) & (u <= c**2))
		factor[j] = 5*a**4 - 6*np.pi*a**3*b + 12*np.pi*a*b*c*np.sqrt(u[j]) + 8*c*u[j]**(3/2.0) - 12*np.pi*a*b*c*np.sqrt(u[j] - a**2) \
						- 8*c*(u[j] - a**2)**(3/2.0) - 12*a*c*u[j]*np.arcsin(a / np.sqrt(u[j]))
		k = np.where((c**2 < u) & (u <= a**2 + c**2))
		factor[k] = 5*a**4 - 6*np.pi*a**3*b + 6*np.pi*a*b*c**2 - c**4 + 6*(np.pi*a*b + c**2)*u[k] + 3*u[k]**2 \
						- 12*np.pi*a*b*c*np.sqrt(u[k] - a**2) - 8*c*(u[k] - a**2)**(3/2) - 4*a*(2*u[k] + c**2)*np.sqrt(u[k] - c**2) \
						+ 12*a*c*u[k]*(np.arccos(c/np.sqrt(u[k])) - np.arcsin(a/np.sqrt(u[k])))

	return prefactor*factor

def h22(u, a, b, c, highRes):
	prefactor = 1.0 / (3*a**2*b**2*c**2)

	if highRes:
		if 0 < u <= a**2: 
			factor = 0
		elif a**2 < u <= a**2 + b**2:
			factor = 3*np.pi*a**2*b*(a + c) - 3*a**4 - 6*np.pi*a*b*c*u**0.5 + 3*(a**2 + np.pi*b*c)*u \
								+ (6*np.pi*a*b*c - 2*(b + 3*c)*a**2 - 4*b*u)*(u - a**2)**0.5 - 6*a*b*u*mp.asin(a/u**0.5)
		elif a**2 + b**2 < u <= a**2 + c**2:
			factor = 3*a**2*b*(np.pi*a - b) - 4*b**4 - 12*a*b*c*mp.asin(b*u**0.5 / ((a**2 + b**2)**0.5*(u - a**2)**0.5)) \
			*u**0.5 - 6*a*c*(a - np.pi*b)*(u - a**2)**0.5 - 6*c*(b**2 - a**2 + 2*a*b*mp.asin(a /(a**2 + b**2)**0.5)) \
						*(u - a**2 - b**2)**0.5 - 6*a*b*(a**2 + b**2)*mp.asin(a/(a**2 + b**2)**0.5) + 6*b*c*(a**2 + u) \
						*mp.asin(b/(u - a**2)**0.5)
		elif a**2 + c**2 < u <= a**2 + b**2 + c**2:
			factor = 3*a**2*(a**2 - b**2 - c**2) - 4*b**4 - 3*a**2*u - 12*a*b*c*(mp.asin(b*u**0.5 / ((a**2 + b**2)**0.5 \
				*(u - a**2)**0.5)) - mp.acos(a*c / ((u - c**2)**0.5*(u - a**2)**0.5)))*u**0.5 \
				+ 2*b*(a**2 + c**2 + 2*u)*(u - a**2 - c**2)**0.5 - 6*c*(b**2 - a**2 \
				+ 2*a*b*mp.asin(a / (a**2 + b**2)**0.5))*(u - a**2 - b**2)**0.5 - 6*a*b*(a**2 + b**2) \
				*mp.asin(a /(a**2 + b**2)**0.5) + 6*b*c*(a**2 + u)*(mp.asin(b /(u - a**2)**0.5) \
				- mp.acos(c /(u - a**2)**0.5)) + 6*a*b*(c**2 + u)*mp.asin(a /(u - c**2)**0.5)
		else:
			factor = 0
	else:
		factor = np.zeros(len(u), dtype=np.float64)
		i = np.where((0 < u) & (u <= a**2))
		factor[i] = 0
		j = np.where((a**2 < u) & (u <= a**2 + b**2))
		factor[j] = 3*np.pi*a**2*b*(a + c) - 3*a**4 - 6*np.pi*a*b*c*np.sqrt(u[j]) + 3*(a**2 + np.pi*b*c)*u[j] \
								+ (6*np.pi*a*b*c - 2*(b + 3*c)*a**2 - 4*b*u[j])*np.sqrt(u[j] - a**2) - 6*a*b*u[j]*np.arcsin(a / np.sqrt(u[j]))
		k = np.where((a**2 + b**2 < u) & (u <= a**2 + c**2))
		factor[k] = 3*a**2*b*(np.pi*a - b) - 4*b**4 - 12*a*b*c*np.arcsin(b*np.sqrt(u[k]) / (np.sqrt(a**2 + b**2)*np.sqrt(u[k] - a**2))) \
						*np.sqrt(u[k]) - 6*a*c*(a - np.pi*b)*np.sqrt(u[k] - a**2) - 6*c*(b**2 - a**2 + 2*a*b*np.arcsin(a / np.sqrt(a**2 + b**2))) \
						*np.sqrt(u[k] - a**2 - b**2) - 6*a*b*(a**2 + b**2)*np.arcsin(a / np.sqrt(a**2 + b**2)) + 6*b*c*(a**2 + u[k]) \
						*np.arcsin(b/np.sqrt(u[k] - a**2))
		l = np.where((a**2 + c**2 < u) & (u <= a**2 + b**2 + c**2))
		factor[l] = 3*a**2*(a**2 - b**2 - c**2) - 4*b**4 - 3*a**2*u[l] - 12*a*b*c*(np.arcsin(b*np.sqrt(u[l]) / (np.sqrt(a**2 + b**2) \
				*np.sqrt(u[l] - a**2))) - np.arccos(a*c / (np.sqrt(u[l] - c**2)*np.sqrt(u[l] - a**2))))*np.sqrt(u[l]) \
				+ 2*b*(a**2 + c**2 + 2*u[l])*np.sqrt(u[l] - a**2 - c**2) - 6*c*(b**2 - a**2 \
				+ 2*a*b*np.arcsin(a / np.sqrt(a**2 + b**2)))*np.sqrt(u[l] - a**2 - b**2) - 6*a*b*(a**2 + b**2) \
				*np.arcsin(a / np.sqrt(a**2 + b**2)) + 6*b*c*(a**2 + u[l])*(np.arcsin(b / np.sqrt(u[l] - a**2)) \
				- np.arccos(c / np.sqrt(u[l] - a**2))) + 6*a*b*(c**2 + u[l])*np.arcsin(a / np.sqrt(u[l] - c**2))
			
	return prefactor*factor

def h32(u, a, b, c, highRes):
	return h22(u, b, a, c, highRes)

def h33(u, a, b, c, highRes):
	prefactor = 1.0 / (6*a**2*b**2*c**2)
	
	if highRes:
		if 0 < u <= b**2:
			factor = 0
		elif b**2 < u <= a**2 + b**2:
			factor = 3*(2*np.pi*a*b + b**2 + u)*(u - b**2) - 4*c*(b**2 + 3*np.pi*a*b + 2*u)*(u - b**2)**0.5
		elif a**2 + b**2 < u <= b**2 + c**2: 
			factor = 3*(a**2 + b**2)**2 - 3*b**4 + 6*np.pi*a**3*b - 4*c*(b**2 + 3*np.pi*a*b + 2*u)*(u - b**2)**0.5 \
						+ 4*c*(a**2 + b**2 + 3*np.pi*a*b + 2*u)*(u - a**2 - b**2)**0.5
		elif b**2 + c**2 < u <= a**2 + b**2 + c**2: 
			factor = 3*(a**2 + b**2)**2 + c**4 + 6*np.pi*a*b*(a**2 + b**2 - c**2) - 6*(np.pi*a*b + c**2)*u - 3*u**2 \
						+ 4*c*(a**2 + b**2 + 3*np.pi*a*b + 2*u)*(u - a**2 - b**2)**0.5
		else:
			factor = 0
	else:
		factor = np.zeros(len(u), dtype=np.float64)
		i = np.where((0 < u) & (u <= b**2))
		factor[i] = 0
		j = np.where((b**2 < u) & (u <= a**2 + b**2))
		factor[j] = 3*(2*np.pi*a*b + b**2 + u[j])*(u[j] - b**2) - 4*c*(b**2 + 3*np.pi*a*b + 2*u[j])*np.sqrt(u[j] - b**2)
		k = np.where((a**2 + b**2 < u) & (u <= b**2 + c**2))
		factor[k] = 3*(a**2 + b**2)**2 - 3*b**4 + 6*np.pi*a**3*b - 4*c*(b**2 + 3*np.pi*a*b + 2*u[k])*np.sqrt(u[k] - b**2) \
						+ 4*c*(a**2 + b**2 + 3*np.pi*a*b + 2*u[k])*np.sqrt(u[k] - a**2 - b**2)
		l = np.where((b**2 + c**2 < u) & (u <= a**2 + b**2 + c**2))
		factor[l] = 3*(a**2 + b**2)**2 + c**4 + 6*np.pi*a*b*(a**2 + b**2 - c**2) - 6*(np.pi*a*b + c**2)*u[l] - 3*u[l]**2 \
						+ 4*c*(a**2 + b**2 + 3*np.pi*a*b + 2*u[l])*np.sqrt(u[l] - a**2 - b**2)
	
	return prefactor*factor

# PDF of u, equation 15
def h(u, a, b, c, highRes):
	return h11(u, a, b, c, highRes) + h12(u, a, b, c, highRes) + h22(u, a, b, c, highRes) + h32(u, a, b, c, highRes) + h33(u, a, b, c, highRes)

# PDF of separations v = u**0.5
def k(v, a, b, c, highRes=True):
	if np.isscalar(v): v = np.array([v])
	if highRes == True: 
		a, b, c = mp.mpf(a), mp.mpf(b), mp.mpf(c)
		k_ = []
		for v_ in v:
			v_ = mp.mpf(v_)
			k_.append(float(2*v_*h(v_**2, a, b, c, highRes)))
	else:
		k_ = 2*v*h(v**2, a, b, c, highRes)

	return k_

# Equation 22 - unit cube (just scale a,b,c for an arbitrary cube and include the Jacobian) 
def kCube(r, a, highRes=False):
	rMax = 3**0.5 * a
	if highRes:
		pdf = []
		if np.isscalar(r): r = [r]
		for r_ in r:
			r_ = mp.mpf(r_)
			# Rescale a, b, c to find PDF for cube that isn't necessarily unit lengthed
			v = r_ * 3**0.5 / rMax 
			if (0 <= v) and (v <= 1):
				pdf_ = v**2 * (4*np.pi - 6*np.pi*v + 8*v**2 - v**3)
			elif (1 < v) and (v <= 2**0.5):
				pdf_ = (6*np.pi - 1)*v - 8*np.pi*v**2 + 6*v**3 + 2*v**5 + 24*v**3*mp.atan((v**2 - 1)**0.5) \
							- 8*v * (1 + 2*v**2) * (v**2 - 1)**0.5
			elif (2**0.5 < v) and (v <= 3**0.5):
				pdf_ = (6*np.pi - 5)*v - 8*np.pi*v**2 + 6*(np.pi - 1)*v**3 - v**5 + 8*v * (1 + v**2) * (v**2 - 2)**0.5 \
							- 24*v * (1 + v**2) * mp.atan((v**2 - 2)**0.5) + 24*v**2 * mp.atan(v * (v**2 - 2)**0.5)
			else:
				pdf_ = 0.0
			pdf.append(float(pdf_ * 3**0.5 / rMax)) # Jacobian
		pdf = np.array(pdf)
	else:
		v = r * 3**0.5 / rMax
		if np.isscalar(v): v = np.array([v])
		pdf = np.zeros(len(v), dtype=np.float64)
		i = np.where((0 < v) & (v <= 1))
		pdf[i] = v[i]**2 * (4*np.pi - 6*np.pi*v[i] + 8*v[i]**2 - v[i]**3)
		j = np.where((1 < v) & (v <= 2**0.5))
		pdf[j] = (6*np.pi - 1)*v[j] - 8*np.pi*v[j]**2 + 6*v[j]**3 + 2*v[j]**5 + 24*v[j]**3*np.arctan((v[j]**2 - 1)**0.5) \
					- 8*v[j] * (1 + 2*v[j]**2) * (v[j]**2 - 1)**0.5
		l = np.where((2**0.5 < v) & (v <= 3**0.5))
		pdf[l] = (6*np.pi - 5)*v[l] - 8*np.pi*v[l]**2 + 6*(np.pi - 1)*v[l]**3 - v[l]**5 + 8*v[l] * (1 + v[l]**2) * (v[l]**2 - 2)**0.5 \
					- 24*v[l] * (1 + v[l]**2) * np.arctan((v[l]**2 - 2)**0.5) + 24*v[l]**2 * np.arctan(v[l] * (v[l]**2 - 2)**0.5)
	
		pdf *=  3**0.5 / rMax # Jacobian

	return pdf 
#######################################################################################################################


###################################################################################################
# Functions for the user to directly call
###################################################################################################
def figure3(minV = 1.e-5, vN=100, highRes=False): # From J. Philip 2007
	if rank == 0:
		sidelengths = (4, 5, 6)
		maxV = (sidelengths[0]**2 + sidelengths[1]**2 + sidelengths[2]**2)**0.5
		v_ = np.linspace(minV, maxV, vN)
		
		k_ = k(v_, *sidelengths, highRes=highRes)

		helper.plot(v_, k_, xlabel='v', ylabel='k', filename='boxPDFfigure3', log=False)

	return None

def computePDF(a, b, c, rN=1000, highRes=False, Norders_minMax=10, errorThresh=0.1):
	# a 				(float or int) 	box sidelength
	# b 				(float or int) 	box sidelength
	# c 				(float or int) 	box sidelength
	# rN 				(int)			number of pair separations for computing the PDF
	# highRes 			(bool)			use mpmath where we can control the floating point precision
	# Norders_minMax	(int) 			number of orders of magnitude between the minimum and maximum pair separation
	# errorThresh 		(float or int)	check that the pdf is normalized within errorThresh percent

	# returns : r (np array) pair separations, pdf (np array) probability density of pair separations 

	if rank == 0:
		a, b, c = float(a), float(b), float(c)
		rMax = 1.01*(a**2 + b**2 + c**2)**0.5
		rMin = 10**(np.log10(rMax) - Norders_minMax)
		r = np.logspace(np.log10(rMin), np.log10(rMax), rN) 
		if a == b == c: # Cube
			pdf = kCube(r, a)
		else:
			a, b, c = np.sort([a, b, c]) # The paper states that a <= b <= c
			rectRAB_list, rectPDF_list = [], []

			# Compute pdf and check that it is normalized within errorThresh percent
			if not highRes:
				pdf = k(r, a, b, c, highRes)
				error = 100*abs(1-integrate.simps(pdf, r))
			if highRes or ((not highRes) and (error > errorThresh)):
				pdf = k(r, a, b, c, highRes=True) # Uses mpmath, slower
				error = 100*abs(1-integrate.simps(pdf, r))
			
			if error > errorThresh: # Higher resolution didn't work. Maybe check that rMax and rMin well-chosen
				print('Warning, box PDF normalization deviation (%)', round(error, 3), 'is greater than error threshold'); sys.stdout.flush()
			pdf = np.array(pdf)

		print('Box sidelengths', a, b, c, '[length units]')
		print('Box PDF has peak at approx width/2 [length units] (for narrow geometries)')
		print('Box volume V=' + str(a*b*c) + ' [length units]^3'); sys.stdout.flush()

	else:
		r, pdf = np.empty(rN, dtype=np.float64), np.empty(rN, dtype=np.float64) # Initialize

	if size > 1: 
		comm.Bcast(r, root=0)
		comm.Bcast(pdf, root=0)

	return r, pdf
###################################################################################################

