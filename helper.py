# Helper functions that we use in other scripts in this directory
# __main__ usually refers to example.py 

from __main__ import *
# From __main__:
# rank 		(int)	the processor label (starts from 0)
# savefigs	(bool)	save figures to current directory
# showfigs	(bool)	show figures - this doesn't always work because the matplotlib backend can be non-interactive

def arcminToRadians(openingAngle):
	polarAngle = openingAngle/2.0
	return polarAngle/60.0 * (2*np.pi/360)

def z_to_comoving(zMax, Omega_cdm, Omega_b, zMin=0): # Matter, Lambda only components (LambdaCDM) (Mpc/h)
	# zMax 		(int or float or list or np array) 	maximum redshift
	# Omega_cdm (int or float)						DM density parameter
	# Omega_b 	(int or float)						baryon density parameter
	# zMax 		(int or float) 						minimum redshift

	# returns: chi (float if input is a scalar otherwise a np array) comoving distance

	c = 299792.458 # km/s
	
	def Hubble(z): 
		E = (Omega_cdm + Omega_b) * (1.0 + z)**3 + (1 - Omega_cdm - Omega_b) 
		return 100 * E**0.5

	def dchi_dz(z):
		return c/Hubble(z)

	if np.isscalar(zMax) == True:
		chi = integrate.quad(dchi_dz, zMin, zMax)[0]
	else:
		chi = np.empty(len(zMax))
		for i, z_ in enumerate(zMax):
			chi[i] = integrate.quad(dchi_dz, zMin, z_)[0]

	return chi

def comoving_to_z(new_r, Omega_cdm, Omega_b):
    z = np.hstack((0, np.logspace(-10, 4, 1000)))
    r = z_to_comoving(z, Omega_cdm, Omega_b)
    
    return logInterp(float(new_r), r, z)

def boxSidelengths(rMin, rMax, openingAngle, Omega_cdm, Omega_b):
	depth = rMax - rMin
	zMin, zMax = comoving_to_z(rMin, Omega_cdm, Omega_b), comoving_to_z(rMax, Omega_cdm, Omega_b)
	zMid = (zMax + zMin)/2.0
	rMid = z_to_comoving(zMid, Omega_cdm, Omega_b) # Not the average of rMax and rMin!
	width = 2 * np.tan(arcminToRadians(openingAngle)) * rMid

	return width, width, depth

def logInterp(newX, x, f): # Interpolate/extrapolate in log-log space
	# newX	(int or float or list or np array) 	value to extrapolate to
	# x 	(list or np array) 					function domain x
	# f 	(list or np array) 					function f(x)

	# returns (np array) interpolated function

	newX, x, f = np.array(newX), np.array(x), np.array(f)

	# Shift non-positive functions so we can take the log without divergences
	min_x = min(np.min(x), np.min(newX))
	min_y = np.min(f)
	if min_x > 0.0:
		trans_x = 0.0
	else:
		trans_x = abs(min_x) + 1.e-5
	if min_y > 0.0:
		trans_y = 0.0
	else:
		trans_y = abs(min_y) + 1.e-5
	f += trans_y
	x += trans_x
	newX += trans_x

	f_interp = interp1d(np.log10(x), np.log10(f), kind='linear', fill_value='extrapolate')
	f_interp = 10**(f_interp(np.log10(newX)))
	f_interp -= trans_y # Now translate back to the original function
	
	return f_interp

def plot(x, y, xlabel, ylabel, filename, log=True, abs_=True):
	# x 		(list or np array) 	function domain
	# y 		(list or np array) 	function y(x)
	# xlabel 	(str) 				x-axis label
	# ylabel 	(str) 				y-axis label
	# filename 	(str) 				for saving the plot if savefigs==True
	# log 		(bool) 				log-scale the x- and y-axes
	# abs_ 		(bool) 				take the absolute value of the y values (only done if log==True)

	if rank == 0:
		if log:
			if abs_:
				x, y = np.array(x), np.array(y)
				i, j = np.where(y >= 0)[0], np.where(y < 0)[0]
				x_pos, y_pos = x[i], y[i] 
				x_neg, y_neg = x[j], y[j] 
				plt.scatter(x_pos, y_pos, s=1, color='b')
				if len(j) == 0:
					plt.title('(no negative values found)')
				else:
					plt.scatter(x_neg, np.abs(y_neg), s=1, color='g', label='negative values')
					plt.legend()
				plt.xscale('log')
				plt.yscale('log')
			else:
				plt.loglog(x, y)
		else:
			plt.plot(x, y)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if savefigs: plt.savefig(filename + '.png', dpi=400)
		if showfigs: plt.show()
		plt.close()

	return None 