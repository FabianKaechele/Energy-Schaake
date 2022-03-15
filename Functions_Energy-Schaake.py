
import numpy as np
import scipy.stats as st
from numpy import linalg as LA


def density_forecast_nonparam(Yp, sigma, _, rankmatrix, errordist_normed):
    """creates a density forecast for Yp with Schaake Schuffle 

           Parameters
           ----------
           Yp: numpy.array
                    24-dimensional array with point-predictions of day ahead prices

           sigma: numpy.array
                    Variance prediction for each hour

           _ :

           rankmatrix: numpy.array
                    Matrix with rank positions of forecast samples

           errordist_normed:  numpy.array
                    Realized normed prediction errors 

           Returns
           -------
           newdataarray:   numpy.array
                    Array containing the density predictions of day ahead price

           """
    # initialize
    nzero=np.size(rankmatrix,axis=0)
    sqrtsigma = np.sqrt(sigma)
    Yt = np.zeros(shape=(nzero, 24))
    u_new = np.arange(1, nzero + 1) / (nzero + 1)
    std_error = np.zeros(shape=(nzero, 24))
    
    # Create sorted array of (correctly-)scaled univariate marginals
    for h in range(24):
        helper = np.sort(errordist_normed[:, h])
        std_error_pos = np.array(np.floor(u_new * np.size(errordist_normed, axis=0)), dtype='int')  
        std_error[:, h] = helper[std_error_pos]
    for i in range(nzero):
        Yt[i, :] = std_error[i, :] * sqrtsigma + Yp

    # Order marginals according to rank-matrix
    newdataarray = np.zeros(shape=(nzero, 24))
    for col in range(24):
        for i in range(0, nzero):
            help = int(rankmatrix[i, col] - 1)
            newdataarray[i, col] = Yt[help, col]

    return newdataarray


def density_forecast_param(Yp, sigma, _, rankmatrix, errordist_normed, dof):
    """creates a density forecast for Yp with Schaake Schuffle

            Parameters
           ----------
           Yp: numpy.array
                    24-dimensional array with point-predictions of day ahead prices

           sigma: numpy.array
                    Variance prediction for each hour

           _ :

           rankmatrix: numpy.array
                    Matrix with rank positions of forecast samples

           errordist_normed:  numpy.array
                    Realized normed prediction errors 
                    
           dof: int
                    Degrees of Freedom of parametric margins
                    0:  Normal distribution
                    >0: t-distribution 
           

           Returns
           -------
           newdataarray:   numpy.array
                       Array containing the density predictions of day ahead price

           

           """
    # Initialize
    errordist=errordist_normed.copy()
    nzero=np.size(rankmatrix,axis=0)
    n_sample=np.size(errordist, axis=0)
    sqrtsigma = np.sqrt(sigma)
    
    # 
    for h in range(24):
        # Assume Normal distribution for dof==0
        if dof[0]==0:
            errordist[:, h]=np.linspace(st.norm(Yp[0, h], sqrtsigma[h]).ppf(1 / (n_sample + 1)), st.norm(Yp[0, h], sqrtsigma[h]).ppf(n_sample / (n_sample + 1)), n_sample)
        # Assume t-distribution with given degrees of freedom
        else:
            errordist[:, h] = np.linspace(st.t(loc=Yp[0, h], scale=sqrtsigma[h],df=dof[h]).ppf(1 / (n_sample + 1)),
                                          st.t(loc=Yp[0, h], scale=sqrtsigma[h],df=dof[h]).ppf(n_sample / (n_sample + 1)), n_sample)

    Yt = np.zeros(shape=(nzero, 24))
    u_new = np.arange(1, nzero + 1) / (nzero + 1)
    std_error = np.zeros(shape=(nzero, 24))
    for h in range(24):
        helper = np.sort(errordist[:, h])
        std_error_pos = np.array(np.floor(u_new * np.size(errordist, axis=0)), dtype='int') 
        std_error[:, h] = helper[std_error_pos]

    for i in range(nzero):
        Yt[i, :] = std_error[i, :] 

    # order newdata according to rank-matrix
    newdataarray = np.zeros(shape=(nzero, 24))
    for col in range(24):
        for i in range(0, nzero):
            help = int(rankmatrix[i, col] - 1)
            newdataarray[i, col] = Yt[help, col]

    return newdataarray


def get_rankmatrix(data,param):
    """creates the rankmatrix needed for forecast construction

            Parameters
           ----------
           data : numpy.array
                    errors of last n days, where the rankmatrix should be retrieved

           param: boolean
                    True: Parametric Dependence strucutre (Gaussian)
                    False: Non-Parametric dependece strucutre
           Returns
           -------
           rankmatrix:   numpy.array
                      matrix encoding the dependence structure 


           """
    # initialize
    rankmatrix = np.zeros(shape=(np.size(data, axis=0), 24))
    # Rank data
    for col in range(data.shape[1]):
        rankmatrix[:, col] = st.rankdata(data[:, col])

    if param==True:
        cov = np.corrcoef(rankmatrix.astype(float), rowvar=False, bias=True)
        ens = np.random.multivariate_normal(np.zeros(24), cov, np.size(data, axis=0))
        rankmatrix = np.zeros(shape=(np.size(ens, axis=0), 24))
        for col in range(ens.shape[1]):
            rankmatrix[:, col] = st.rankdata(ens[:, col])

    return rankmatrix

    