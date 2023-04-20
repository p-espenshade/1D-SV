###################################################################################################
# These flags affect the libraries that are imported
parallel = False # If False: python example.py. If True (e.g.): mpiexec -n 4 python example.py
useNbodykit = False # Used to compute the 2-point correlation (otherwise load it from data)
###################################################################################################
# Plotting
savefigs = True # Saves figures to current directory
showfigs = False # Shows the figures - this doesn't always work because the matplotlib backend can be non-interactive
###################################################################################################


###################################################################################################
# Import modules
###################################################################################################
import sys 
import matplotlib.pyplot as plt
import numpy as np 
from scipy import integrate
from scipy.interpolate import interp1d
from mpmath import mp

if parallel:
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
else:
	rank, size = 0, 1
if rank == 0: 
	parallelStr = 'parallel' if parallel else 'serial'
	print('Running in', parallelStr, 'with', size, 'processor(s)'); sys.stdout.flush()

if useNbodykit:	from nbodykit.lab import cosmology

import helper
import conePDF, boxPDF, sv, morePDFs
###################################################################################################


###################################################################################################
# Parameters
###################################################################################################
# Cosmology (used to convert comoving distances to redshift - we need the average redshift for the box geometry)
Omega_cdm = 0.27 # (float or int)
Omega_b = 0.049 # (float or int)

sampleN = 10**8 # (int) For cone PDF - this is the sampled number of pairs. Controls the precision

# Pair separations will be computed for rN separations 
rN = 1000 # (int)
Norders_minMax = 10 # (int) Number of orders of magnitude between the minimum and maximum separations. The max separation is fixed by the survey geometry

# Geometry
openingAngle = 1 # (int or float) Arcminutes
rMin, rMax = 0, 1300 # (int or float). Can be in any length units e.g. Mpc/h. For spherical coordinate r, the cone extends from rMin <= r < rMax
rUnits = 'Mpc/h' # (str) Only used for plotting
# rMin = helper.z_to_comoving(0.1, Omega_cdm, Omega_b) # Convert redshift to comoving distance in Mpc/h
# rMax = helper.z_to_comoving(0.2, Omega_cdm, Omega_b)

corrSource = 'corr_nonlinear.txt' # (str) 2-point correlation filename. If useNbodykit==True, corrSource is ignored
# corrSource = 'corr_linear.txt' 
###################################################################################################


###################################################################################################
# Compute the probability density functions (PDFs) as a function of pair separation r
###################################################################################################
r_cone, pdf_cone = conePDF.computePDF(sampleN=sampleN, openingAngle=openingAngle, rMin=rMin, rMax=rMax, \
									rN=rN, Norders_minMax=Norders_minMax)

a, b, c = helper.boxSidelengths(rMin, rMax, openingAngle, Omega_cdm, Omega_b)
r_box, pdf_box = boxPDF.computePDF(a, b, c, rN=rN, Norders_minMax=Norders_minMax)

# Note that the line-of-sight (LOS), extends from rMin=0 to rMax, and so does the sphere.
r_los, pdf_los = morePDFs.losPDF(rMax, rN=rN, Norders_minMax=Norders_minMax)
r_sphere, pdf_sphere = morePDFs.spherePDF(rMax, rN=rN, Norders_minMax=Norders_minMax)

helper.plot(r_cone, pdf_cone, xlabel='r [' + rUnits +']', ylabel='|PDF(r)| [' + rUnits + ']^-1', filename='conePDF', abs_=True)
helper.plot(r_box, pdf_box, xlabel='r [' + rUnits +']', ylabel='|PDF(r)| [' + rUnits + ']^-1', filename='boxPDF', abs_=True)
helper.plot(r_los, pdf_los, xlabel='r [' + rUnits +']', ylabel='|PDF(r)| [' + rUnits + ']^-1', filename='losPDF', abs_=True)
helper.plot(r_sphere, pdf_sphere, xlabel='r [' + rUnits +']', ylabel='|PDF(r)| [' + rUnits + ']^-1', filename='spherePDF', abs_=True)
###################################################################################################


###################################################################################################
# Load the 2-point correlation or use nbodykit to compute it
###################################################################################################
# Note that the slope of the 2-point correlation on small scales can greatly affect the sample variance in some cases
# because the 2-point correlation is extrapolated to very small r values when integrating for the sample variance
###################################################################################################
if (not useNbodykit) and (rank == 0):
	print('Loading correlation from', corrSource); sys.stdout.flush()
	data = np.loadtxt(corrSource)
	r_corr, corr = data.T
	r_corr, corr = np.array(r_corr), np.array(corr)
	corrN = len(corr)

elif useNbodykit and (rank == 0):
	linear = False
	linearStr = 'linear' if linear else 'nonlinear'
	print('Using nbodykit to compute the', linearStr, '2-point correlation'); sys.stdout.flush()
	cosmo = cosmology.Planck15
	cosmo = cosmo.clone(Omega_cdm=Omega_cdm, Omega_b=Omega_b, h=0.67, n_s=0.97, A_s=np.exp(3.0)/(1.e10))
	zMin, zMax = helper.comoving_to_z(rMin, Omega_cdm, Omega_b), helper.comoving_to_z(rMax, Omega_cdm, Omega_b) # Assumes rMin, rMax are in Mpc/h
	zMid = (zMax + zMin)/2.0
	# zMid = 0.0
	if linear == True:
		P = cosmology.LinearPower(cosmo, redshift=zMid, transfer='CLASS')
	else:
		P = cosmology.HalofitPower(cosmo, redshift=zMid)
	corrN = 1000
	r_corr = np.logspace(np.log10(0.05), np.log10(300), corrN) # rMin arbitrarily chosen here - the user should carefully consider what value to use here
	corr = cosmology.CorrelationFunction(P)(r_corr)
	# np.savetxt('corr_' + linearStr + '.txt', np.vstack((r_corr, corr)).T, header='r [Mpc/h],\tcorr(r)\t\t(z='+str(zMid)+')')

if size > 1:
	if rank > 0: corrN = None # Initialize
	corrN = comm.bcast(corrN, root=0)

	if rank > 0: 
		r_corr, corr = np.empty(corrN, dtype=np.float64), np.empty(corrN, dtype=np.float64) # Initialize
	comm.Bcast(r_corr, root=0); comm.Bcast(corr, root=0)

helper.plot(r_corr, corr, xlabel='r [' + rUnits +']', ylabel='|corr(r)|', filename='corr', abs_=True)
###################################################################################################


###################################################################################################
# Compute the variance (square rooted) using the 2-point correlation and the PDF for the survey geometry
###################################################################################################
sv_cone, integrand_cone, r_cone_sv = sv.computeSV(r_corr, corr, r_cone, pdf_cone, extremeExtrap=False, printStr='Cone') # Integrand = PDF(r)*corr(r)
sv_box, integrand_box, r_box_sv = sv.computeSV(r_corr, corr, r_box, pdf_box, extremeExtrap=False, printStr='Box')
sv_los, integrand_los, r_los_sv = sv.computeSV(r_corr, corr, r_los, pdf_los, extremeExtrap=False, printStr='LOS') 
sv_sphere, integrand_sphere, r_sphere_sv = sv.computeSV(r_corr, corr, r_sphere, pdf_sphere, extremeExtrap=False, printStr='Sphere') 

helper.plot(r_cone_sv, r_cone_sv*integrand_cone, xlabel='r [' + rUnits +']', ylabel='|r * PDF(r) * corr(r)|', filename='svIntegrandCone', abs_=True)
helper.plot(r_box_sv, r_box_sv*integrand_box, xlabel='r [' + rUnits +']', ylabel='|r * PDF(r) * corr(r)|', filename='svIntegrandBox', abs_=True)
helper.plot(r_los_sv, r_los_sv*integrand_los, xlabel='r [' + rUnits +']', ylabel='|r * PDF(r) * corr(r)|', filename='svIntegrandLOS', abs_=True)
helper.plot(r_sphere_sv, r_sphere_sv*integrand_sphere, xlabel='r [' + rUnits +']', ylabel='|r * PDF(r) * corr(r)|', filename='svIntegrandSphere', abs_=True)
###################################################################################################

