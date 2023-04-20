# Computes probability distribution (PDF) of pair separations in a cone (with a rounded top, but it is straightforward to alter this code for a flat top)
# Note that this code can be used for the case that the cone becomes a sphere, but it's recommended to instead use the known analytical solution
# Also, it is easy to alter this code to estimate the PDF for other geometries via rejection sampling. Simply generate the pair distribution for a sphere
# 		then cut away the parts of the sphere (it's sculpting, really) to form your desired shape

from __main__ import *

# From __main__:
# size (int) the number of processors
# rank (int) the processor label (starts from 0)
# helper.py

def separationNumberCount(sampleN, openingAngle, rMin, rMax, rEdges, seed=42):
	# sampleN 		(int) 				number of pairs to generate in the survey
	# openingAngle 	(float or int)	 	opening angle of the cone = 2 * polar angle
	# rMin 			(float or int) 		for spherical coordinate r, r >= rMin in the cone
	# rMax 			(float or int) 		r < rMax
	# rEdges 		(list or np array) 	bin edges for pair separations
	# seed 			(int) 				random seed

	# returns : counts (np array of floats) the number count of pair separations in each bin

	np.random.seed(seed)
	singlePointsN = int(sampleN**0.5) # This is the number of ra or rb points to be generated
	cosAlpha = np.cos(helper.arcminToRadians(openingAngle))

	# Generate 2 uniform distributions of points in a cone (ra, rb)
	ra = np.random.uniform(rMin**3, rMax**3, singlePointsN)
	np.power(ra, 1/3.0, out=ra)
	rb = np.random.uniform(rMin**3, rMax**3, singlePointsN)
	np.power(rb, 1/3.0, out=rb)
	cosA = np.random.uniform(cosAlpha, 1.0, singlePointsN)
	cosB = np.random.uniform(cosAlpha, 1.0, singlePointsN)
	phiA = np.random.uniform(0, 2*np.pi, singlePointsN)
	phiB = np.random.uniform(0, 2*np.pi, singlePointsN)
	sinA, sinB = (1 - cosA**2)**0.5, (1 - cosB**2)**0.5

	# Compute the distance between every pair of points (rab)
	rabN = singlePointsN**2
	rab = np.empty(rabN)
	for i in range(singlePointsN):
		cosAlpha = sinA[i]*sinB*np.cos(phiA[i]-phiB) + cosA[i]*cosB # cosAngle between vectors
		rab[i*singlePointsN : (i+1)*singlePointsN] = (ra[i]**2 + rb**2 - 2*ra[i]*rb*cosAlpha)**0.5

	# The probability distribution of separations is approximated by the histogram of separations
	counts, _ = np.histogram(rab, bins=rEdges) 

	return counts.astype(np.float64)

def peakOfPDF(openingAngle, rMax): # For rMin=0 and narrow geometries
	return np.tan(helper.arcminToRadians(openingAngle))*rMax

def coneVolume(openingAngle, rMin, rMax):
	# First compute volume of a flat cone
	baseRadius1 = rMin * np.tan(helper.arcminToRadians(openingAngle))
	baseRadius2 = rMax * np.tan(helper.arcminToRadians(openingAngle))
	volume1 = np.pi*baseRadius1**2 * rMin/3.0
	volume2 = np.pi*baseRadius2**2 * rMax/3.0
	volume = volume2 - volume1

	# Add volume of the rounded top
	volume += np.pi/3 * rMax**3 * (2+np.cos(helper.arcminToRadians(openingAngle))) \
								* (1-np.cos(helper.arcminToRadians(openingAngle)))**2
	
	return volume

def computePDF(sampleN=10**2, openingAngle=1, rMin=0, rMax=1, rN=1000, Norders_minMax=10):
	# sampleN 			(int) 			number of pairs to generate in the survey
	# openingAngle 		(float or int)	opening angle of the cone = 2 * polar angle
	# rMin 				(float or int)	for spherical coordinate r, r >= rMin in the cone
	# rMax 				(float or int) 	r < rMax
	# rN 				(int)			number of bins for the pair separations in computing the PDF
	# Norders_minMax	(int) 			number of orders of magnitude between the minimum and maximum pair separation

	# returns : rMid (np array) pair separations (the average of each bin), pdf (np array) probability density of pair separations 

	np.random.seed(rank)
	seed = np.random.randint(0, int(1e6))
	sampleN = int(sampleN/size) # Split the sampled pairs among the ranks
	maxPairSeparation = 2.02 * rMax # Approximate but usually good enough
	minPairSeparation = 10**(np.log10(rMax) - Norders_minMax)
	rPairEdges = np.logspace(np.log10(minPairSeparation), np.log10(maxPairSeparation), rN)
	
	counts = separationNumberCount(sampleN, openingAngle, rMin, rMax, rPairEdges, seed)
	if size > 1: # Average over the output from each proc
		allCounts = np.empty((size, len(counts)), dtype=np.float64)
		comm.Gather(counts, allCounts, root=0)
		if rank == 0: counts = np.mean(allCounts, axis=0) 

	if rank == 0:
		binSize = rPairEdges[1:] - rPairEdges[:-1]
		counts /= binSize # If the bins are uneven we need to weight by the size of each
		rMid = 10 ** ((np.log10(rPairEdges[1:]) + np.log10(rPairEdges[:-1])) / 2.0) 
		norm = integrate.simps(counts, rMid)
		pdf = counts/norm

		print('Cone openingAngle, rMin, rMax:', openingAngle, rMin, rMax, '[arcmin, length units, length units]')
		print('Cone PDF has peak at approx r=' + str(peakOfPDF(openingAngle, rMax)) + ' [length units] (for narrow geometries)')
		print('Cone volume V=' + str(coneVolume(openingAngle, rMin, rMax)) + ' [length units]^3'); sys.stdout.flush()

	else:
		rMid, pdf = np.empty(rN-1, dtype=np.float64), np.empty(rN-1, dtype=np.float64) # Initialize
	
	if size > 1: comm.Bcast(rMid, root=0), comm.Bcast(pdf, root=0)

	return rMid, pdf

