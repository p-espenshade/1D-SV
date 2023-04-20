# Probability distribution of separations for geometries with known solutions

from __main__ import *
# From __main__:
# size 	(int) 	the number of processors
# rank 	(int)	the processor label (starts from 0)

# Line-of-sight geometry that is constructed by taking a cone geometry in the limit that alpha becomes infinitesimal
def losPDF(R=1.0, rN=1000, Norders_minMax=10, errorThresh=0.01):
	# R 				(float or int) 	line-of-sight depth
	# rN 				(int)			number of pair separations for computing the PDF
	# Norders_minMax	(int) 			number of orders of magnitude between the minimum and maximum pair separation
	# errorThresh 		(float or int)	check that the pdf is normalized within errorThresh percent

	# returns : r (np array) pair separations, pdf (np array) probability density of pair separations 

	if rank == 0:
		print('LOS extends from rMin=0 to rMax=' + str(R) + ' [length units]')
		
		rMax = 1.01*R
		rMin = 10**(np.log10(rMax) - Norders_minMax)
		r = np.logspace(np.log10(rMin), np.log10(rMax), rN)

		pdf = -3*r**5/float(5*R**6) + 6*r**2/float(R**3) - 9*r/float(R**2) + 18/float(5*R)

		error = 100*abs(1-integrate.simps(pdf, r))
		if error > errorThresh:
			print('Warning, LOS PDF normalization deviation (%)', round(error, 3), 'is greater than error threshold'); sys.stdout.flush()

	else:
		r, pdf = np.empty(rN, dtype=np.float64), np.empty(rN, dtype=np.float64) # Initialize
	
	if size > 1: 
		comm.Bcast(r, root=0), comm.Bcast(pdf, root=0)

	return r, pdf 

# Sphere geometry. See the references in, e.g., S.-J. Tu & E. Fischbach (2002) for a derivation
def spherePDF(R=1.0, rN=1000, Norders_minMax=10, errorThresh=0.01):
	# R 				(float or int) 	sphere radius
	# rN 				(int)			number of pair separations for computing the PDF
	# Norders_minMax	(int) 			number of orders of magnitude between the minimum and maximum pair separation
	# errorThresh 		(float or int)	check that the pdf is normalized within errorThresh percent

	# returns : r (np array) pair separations, pdf (np array) probability density of pair separations 

	if rank == 0:
		print('Sphere extends from rMin=0 to rMax=' + str(R) + ' [length units]')
		print('Sphere volume V=' + str(4/3.0*np.pi*R**3) + ' [length units]^3')

		rMax = 2.02*R
		rMin = 10**(np.log10(rMax) - Norders_minMax)
		r = np.logspace(np.log10(rMin), np.log10(rMax), rN)

		pdf = 3*r**5/float(16*R**6) - 9*r**3/float(4*R**4) + 3*r**2/float(R**3)

		error = 100*abs(1-integrate.simps(pdf, r))
		if error > errorThresh:
			print('Warning, sphere PDF normalization deviation (%)', round(error, 3), 'is greater than error threshold'); sys.stdout.flush()

	else:
		r, pdf = np.empty(rN, dtype=np.float64), np.empty(rN, dtype=np.float64) # Initialize
	
	if size > 1: 
		comm.Bcast(r, root=0), comm.Bcast(pdf, root=0)

	return r, pdf 