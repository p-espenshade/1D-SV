# Compute the sample variance for a given distribution of pair separations and a 2-point correlation
# __main__ usually refers to example.py 

from __main__ import *

# It can be useful to switch extremeExtrap to True and False to check that the sample variance hardly changes (assuming the PDF and corr are linear in log-space)
def computeSV(r_corr, corr, r_pdf, pdf, extremeExtrap=False, printStr=''):
	# r_corr 		(list or np array) 	separations for the 2-point correlation
	# corr 			(list or np array) 	2-point correlation
	# r_pdf 		(list or np array) 	separations for the pdf
	# pdf 			(list or np array) 	probability density function
	# extremeExtrap (bool)				extrapolate the pdf and correlation to extreme values
	# printStr 		(str)				Optional string with geometry to print

	# returns : sample variance (float), integrand used to compute the sv (np array), separations (np array)
	
	if rank == 0:
		r_corr, corr, r_pdf, pdf = np.array(r_corr), np.array(corr), np.array(r_pdf), np.array(pdf)
		new_r = np.sort(np.unique(np.hstack((r_corr, r_pdf)))) # Combine r bins from corr and pdf
		
		if extremeExtrap:
			rMin_extrap = 1.e-10 # Somewhat arbitrary - change based on your r values
			rMax_extrap = 1.e10
			if rMin_extrap < np.min(new_r):
				new_r = np.hstack((rMin_extrap, new_r))
			if rMax_extrap > np.max(new_r):
				new_r = np.hstack((new_r, rMax_extrap))
		rN = len(new_r)

		corr = helper.logInterp(new_r, r_corr, corr)
		pdf = helper.logInterp(new_r, r_pdf, pdf)

		# Normalize the PDF to the new bins (but this step should not change the PDF significantly) 
		norm = integrate.simps(pdf, new_r)
		pdf = pdf/norm

		integrand = corr*pdf
		# Compute the sample variance (technically the standard deviation - sigma not sigma squared)
		sv_ = integrate.simps(integrand, new_r)**0.5 
		
		if (printStr == ''):  
			print('Sample variance (sigma)', sv_)
		else:
			print(printStr + ' sample variance (sigma)', sv_)
	
	if size > 1:
		if rank > 0: rN = None # Initialize
		rN = comm.bcast(rN, root=0)

		if rank > 0:
			sv_, integrand, new_r = None, np.empty(rN, dtype=np.float64), np.empty(rN, dtype=np.float64) # Initialize
		sv_ = comm.bcast(sv_, root=0)
		comm.Bcast(integrand, root=0)
		comm.Bcast(new_r, root=0)

	return sv_, integrand, new_r

