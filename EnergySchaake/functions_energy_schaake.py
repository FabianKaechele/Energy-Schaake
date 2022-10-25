import numpy as np
import scipy.stats as st
import warnings
from rpy2 import robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
# utils = importr('utils')
# utils.install_packages('rugarch')

def learn_multivariate_density(yp, errors, timeseries_treatment=True, lenght_error_learning, length_dependence_learning,
                               param_margin=False, param_dependence=False):
    """ learns multivariate density from series of forecasting errors
    Parameters
           ----------
           yp: numpy.array (1xd)
                    24-dimensional array with point-predictions of day ahead prices
           errors: numpy.array (nxd)
                    24-dimensional with historoc errors of day ahead prices from point forecasting model
           timeseries_treatment: booloean
                    True to fit AR(1)-GARCH(1,1) model for error standardization
           lenght_error_learning: integer
                    number days used for learning univariate marginal distributions
           length_dependence_learning: integer
                    number days used for learning univariate marginal distributions

           Returns
           -------
           dens:   numpy.array
                    """
    # Sanity checks
    len_given_errors = np.size(errors, axis=0)

    if np.size(errors, axis=0) < lenght_error_learning:
        raise ValueError('Given error series is shorter then specified number of errors for error learning!')

    if lenght_error_learning < length_dependence_learning:
        raise ValueError('Lenght_error_learning must be at least of same size then length_dependence_learning!')

    # Time Series Specifications
    if timeseries_treatment == True:
        length_ts_fitting = 365
        if length_ts_fitting > np.size(errors, axis=0):
            warnings.warn("No full year of historic errors is used for time series fitting!")
            length_ts_fitting = np.size(errors, axis=0)
        if np.size(errors, axis=0) < 90:
            raise ValueError('Not enough historic errors for time series standardization!')

        bias = np.zeros(24)
        dof = np.zeros(24)
        sigma = np.zeros(24)
        bias_hist = np.zeros((length_ts_fitting, 24))
        sigma_hist = np.zeros((length_ts_fitting, 24))

        for a in range(24):
            model_archgarchr = r_garch_forecast_t(errors, p=1, q=0, arch_p=1,
                                                  arch_q=1, dist='norm')
            bias[a] = model_archgarchr['Mu']
            sigma[a] = model_archgarchr['Sigma']
            dof[a] = model_archgarchr['dof']
            bias_hist[:, a] = model_archgarchr['Mu_hist'].flatten()
            sigma_hist[:, a] = model_archgarchr['Sigma_hist'].flatten()

        error_std = (errors[-lenght_error_learning:, :] - bias_hist[-lenght_error_learning:]) / sigma_hist[
                                                                                                -lenght_error_learning:]

    # Non-standardized Errors (Raw Error)
    else:
        error_std = errors[-lenght_error_learning:, :]

    # Dependence Learning
    data_dependence = error_std[-length_dependence_learning:, :].copy()
    rankmatrix = get_rankmatrix(data_dependence, param_dependence)

    # Forecast
    if param_margin == False:
        density = density_forecast_nonparam(yp, sigma ** 2, rankmatrix, error_std)
    if param_margin == True:
        density = density_forecast_param(yp, sigma ** 2, rankmatrix, errordist_std, dof)

    return density

    def r_garch_forecast_t(y: np.ndarray, p: int, q: int, arch_p: int, arch_q: int, dist: str = 'norm'):
        """ Uses the R function ugarchestimate to (in-sample) forecast timeseries ARMA-GARCH model with estimation
        on whole data
            Parameters
            ----------
            y: np.ndarray
                Time series values to fit
            p: int
                AR-order of ARMA model
            q: int
                MA-order of ARMA model
            arch_p: int
                AR-order of GARCH model
            arch_q: int
                order of GARCH model
            dist: str
                Innovations distribution (either 'norm' or 'std' for student's t distribution)
            refit_window: str
                'recursive' if estimation window should be increasing, 'moving' for fixed window length
            window_size

            Returns
            -------
            dict
                Dict with forecasts/realisationen: Mu (predicted mean), Sigma (Predicted variance), PIT (CDF-transform of residuals),
                    Resid (Unexplained error), Resid_std (Resid / Sigma)
        """
        model, rugarch = _get_rugarch_model(p=p, q=q, arch_p=arch_p, arch_q=arch_q,
                                            dist=dist)
        modelfit = rugarch.ugarchfit(spec=model, data=y, solver='hybrid')
        sigma_hist = np.asarray(rugarch.sigma(modelfit))
        mu_hist = np.asarray(rugarch.fitted(modelfit))
        forecast = rugarch.ugarchforecast(modelfit)
        sigma = np.asarray(rugarch.sigma(forecast))[0]
        mu = np.asarray(rugarch.fitted(forecast))[0]

        if dist == 'std':
            dof = np.asarray(rugarch.coef(modelfit))[6]
        else:
            dof = 0
        return {'Mu_hist': mu_hist, 'Sigma_hist': sigma_hist, 'Mu': mu, 'Sigma': sigma, 'dof': dof}

    def density_forecast_nonparam(yp, sigma, rankmatrix, errordist_normed):
        """creates a density forecast for yp with Schaake Schuffle
               Parameters
               ----------
               yp: numpy.array
                        24-dimensional array with point-predictions of day ahead prices
               sigma: numpy.array
                        Variance prediction for each hour
               rankmatrix: numpy.array
                        Matrix with rank positions of forecast samples
               errordist_normed:  numpy.array
                        Realized normed prediction errors
               Returns
               -------
               newdataarray:   numpy.array
                        Array containing the density predictions of day ahead price
               """
        # Initialize
        nzero = np.size(rankmatrix, axis=0)
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
            Yt[i, :] = std_error[i, :] * sqrtsigma + yp

        # Order marginals according to rank-matrix
        newdataarray = np.zeros(shape=(nzero, 24))
        for col in range(24):
            for i in range(0, nzero):
                help = int(rankmatrix[i, col] - 1)
                newdataarray[i, col] = Yt[help, col]

        return newdataarray

    def density_forecast_param(yp, sigma, rankmatrix, errordist_std, dof=0):
        """creates a density forecast for Yp with Schaake Schuffle and parametric margins
                Parameters
               ----------
               yp: numpy.array
                        24-dimensional array with point-predictions of day ahead prices
               sigma: numpy.array
                        Variance prediction for each hour

               rankmatrix: numpy.array
                        Matrix with rank positions of forecast samples
               errordist_std:  numpy.array
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
        errordist = errordist_std.copy()
        nzero = np.size(rankmatrix, axis=0)
        n_sample = np.size(errordist, axis=0)
        sqrtsigma = np.sqrt(sigma)

        # Get parametric error distribution samples
        for h in range(24):
            # Assume Normal distribution for dof==0
            if dof[0] == 0:
                errordist[:, h] = np.linspace(st.norm(yp[0, h], sqrtsigma[h]).ppf(1 / (n_sample + 1)),
                                              st.norm(yp[0, h], sqrtsigma[h]).ppf(n_sample / (n_sample + 1)), n_sample)
            # Assume t-distribution with given degrees of freedom
            else:
                errordist[:, h] = np.linspace(st.t(loc=yp[0, h], scale=sqrtsigma[h], df=dof[h]).ppf(1 / (n_sample + 1)),
                                              st.t(loc=yp[0, h], scale=sqrtsigma[h], df=dof[h]).ppf(
                                                  n_sample / (n_sample + 1)), n_sample)

        yt = np.zeros(shape=(nzero, 24))
        u_new = np.arange(1, nzero + 1) / (nzero + 1)
        std_error = np.zeros(shape=(nzero, 24))
        for h in range(24):
            helper = np.sort(errordist[:, h])
            std_error_pos = np.array(np.floor(u_new * np.size(errordist, axis=0)), dtype='int')
            std_error[:, h] = helper[std_error_pos]

        for i in range(nzero):
            yt[i, :] = std_error[i, :]

            # Order newdata according to rank-matrix
        newdataarray = np.zeros(shape=(nzero, 24))
        for col in range(24):
            for i in range(0, nzero):
                help = int(rankmatrix[i, col] - 1)
                newdataarray[i, col] = yt[help, col]

        return newdataarray

    def get_rankmatrix(data, param):
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
        # Initialize
        rankmatrix = np.zeros(shape=(np.size(data, axis=0), 24))
        # Rank data
        for col in range(data.shape[1]):
            rankmatrix[:, col] = st.rankdata(data[:, col])

        if param == True:
            cov = np.corrcoef(rankmatrix.astype(float), rowvar=False, bias=True)
            ens = np.random.multivariate_normal(np.zeros(24), cov, np.size(data, axis=0))
            rankmatrix = np.zeros(shape=(np.size(ens, axis=0), 24))
            for col in range(ens.shape[1]):
                rankmatrix[:, col] = st.rankdata(ens[:, col])

        return rankmatrix
