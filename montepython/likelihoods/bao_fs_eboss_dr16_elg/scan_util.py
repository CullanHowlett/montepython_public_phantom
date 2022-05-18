# scan_util.py
#
# This is a module containing subfunctions to read the chi2 tables from eBOSS DR16 ELG results
#
# class:chi2_interpolators
#       - __init__
#       - get_chi2_distances
#

import numpy as np
from scipy.interpolate import RegularGridInterpolator

#####################################################################

#Class to read alpha_t by alpha_p chi2 scans e.g. from BOSS and interpolate.
class chi2_interpolators():
    def __init__(self,scan_locations,transverse_fid,parallel_fid):
        """
        Arguments:
        scan_locations: dictionary of filepaths to the different scans, with
                        keys as scan types.
        transverse_fid: fiducial value of transverse separation used to
                        calculate alpha_t.
        parallel_fid:   fiducial value of parallel separation used to calculate
                        alpha_p.
        """

        #Create a dictionary containing an interpolator for each scan.
        interpolators = {}
        self.limits = {}
        for corr_type in scan_locations:
            scan = np.loadtxt(scan_locations[corr_type])

            #Column numbers in scan for data points.
            ap_index = 1
            at_index = 0
            fsig8_index = 2
            chi2_index = 3

            #Get the alphas and make the scan grid.
            ap = np.array(sorted(set(scan[:,ap_index])))
            at = np.array(sorted(set(scan[:,at_index])))
            fsig8 = np.array(sorted(set(scan[:,fsig8_index])))
            N_ap = ap.shape[0]
            N_at = at.shape[0]
            N_fsig8 = fsig8.shape[0]

            grid = scan[:,chi2_index].reshape((N_at,N_ap,N_fsig8))

            #Make the interpolator (x refers to at, y refers to ap).
            self.limits[corr_type] = [[np.amin(at), np.amax(at)], [np.amin(ap), np.amax(ap)], [np.amin(fsig8), np.amax(fsig8)]]
            interpolators[corr_type] = RegularGridInterpolator((at,ap,fsig8),grid,method="linear")

        #Add the dictionary to the object.
        self.interpolators = interpolators
        self.transverse_fid = transverse_fid
        self.parallel_fid = parallel_fid

        return

    #Function to return the interpolated value of chi2 given distance measures.
    def get_Dchi2_from_bao_fs(self,transverse,parallel,fsigma8,corr_type='cf'):
        """
        Arguments:
        transverse: value of transverse separation to evaluate chi2 for.
        parallel:   value of parallel separation to evaluate chi2 for.
        corr_type:  which scan to interpolate.

        Returns:
        Dchi2:       value of delta chi2
        """

        #Convert distances to alphas.
        at = transverse/self.transverse_fid
        ap = parallel/self.parallel_fid

        #With the new alphas, get the log likelihood.
        at_flag = ((at >= self.limits[corr_type][0][1]) | (at <= self.limits[corr_type][0][0]))
        ap_flag = ((ap >= self.limits[corr_type][1][1]) | (ap <= self.limits[corr_type][1][0]))
        fsigma8_flag = ((fsigma8 >= self.limits[corr_type][2][1]) | (fsigma8 <= self.limits[corr_type][2][0]))
        if np.any([at_flag, ap_flag, fsigma8_flag]):
            Dchi2 = 1.0e-300
        else:
            Dchi2 = self.interpolators[corr_type]((at,ap,fsigma8))

        return Dchi2
