import numpy as np
import scipy.stats as st
import warnings
from numpy import linalg as LA
import properscoring as ps
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

# R Settings
os.environ['R_HOME'] = '/Users/<your user>/anaconda3/envs/env-name/lib/R'
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
numpy2ri.activate()
# Uncomment following two lines after first usage
utils = importr('utils')
utils.install_packages('rugarch')

def learn_multivariate_density(yp, errors, timeseries_treatment=True, lenght_error_learning=90,
                               length_dependence_learning=90,
                               param_margin=False, param_dependence=False, ts_dist='norm'):
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
           param_margin: boolean
                    use parametric margins if True
           param_dependence: boolean
                    use parametric Gaussian copula if True
           ts_dist: str
                    Specify distribution for parametric margin. 'norm' for Gaussian, 'std' for Students t-distribution
           Returns
           -------
           dens:   numpy.array
                    """
    # Sanity checks
    if np.size(errors, axis=0) < lenght_error_learning:
        raise ValueError('Given error series is shorter then specified number of errors for error learning!')

    if lenght_error_learning < length_dependence_learning:
        raise ValueError('Lenght_error_learning must be at least of same size then length_dependence_learning!')

    # Time Series Specifications
    if timeseries_treatment == True:
        length_ts_fitting = 365
        if length_ts_fitting > np.size(errors, axis=0):
            warnings.warn("No full year of historic errors is used for time series fitting!")
        if np.size(errors, axis=0) < 90:
            raise ValueError('Not enough historic errors for time series standardization!')

        bias = np.zeros(24)
        dof = np.zeros(24)
        sigma = np.zeros(24)
        bias_hist = np.zeros((np.size(errors, axis=0), 24))
        sigma_hist = np.zeros((np.size(errors, axis=0), 24))

        for a in range(24):
            # Fit time series for each hour
            model_archgarchr = r_garch_forecast_t(errors[:, a], p=1, q=0, arch_p=1,
                                                  arch_q=1, dist=ts_dist)
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
        sigma = np.ones(24)
        bias = np.zeros(24)
        dof = np.zeros(24)

    # Dependence Learning
    data_dependence = error_std[-length_dependence_learning:, :].copy()
    rankmatrix = get_rankmatrix(data_dependence, param_dependence)

    # Forecast
    if param_margin == False:
        density = density_forecast_nonparam(yp - bias, sigma ** 2, rankmatrix, error_std)
    if param_margin == True:
        density = density_forecast_param(yp - bias, sigma ** 2, rankmatrix, error_std, dof)

    return density, sigma, bias, dof


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


        Returns
        -------
        dict
            Dict with forecasts/realisationen: Mu_hist (historic mu), Sigma_hist (historic Sigma), Mu (predicted mean),
            Sigma (Predicted variance), dof (degrees of freedom)
    """
        for i in range(50):
        # try to solve with different rseeds
        try:
            model, rugarch = _get_rugarch_model(p=p, q=q, arch_p=arch_p, arch_q=arch_q,
                                            dist=dist)
            modelfit = rugarch.ugarchfit(spec=model, data=y, solver='hybrid', rseed=i)
            sigma_hist = np.asarray(rugarch.sigma(modelfit))
            mu_hist = np.asarray(rugarch.fitted(modelfit))
            forecast = rugarch.ugarchforecast(modelfit)
            sigma = np.asarray(rugarch.sigma(forecast))[0]
            mu = np.asarray(rugarch.fitted(forecast))[0]

            if dist == 'std':
                dof = np.asarray(rugarch.coef(modelfit))[6]
            else:
                dof = 0
            break
        except:
            print('Try again with different r-seed for optimizer!')
            pass
    return {'Mu_hist': mu_hist, 'Sigma_hist': sigma_hist, 'Mu': mu, 'Sigma': sigma, 'dof': dof}


def density_forecast_nonparam(yp, sigma, rankmatrix, errordist_normed):
    """creates a density forecast for yp with Schaake Schuffle and nonparametric margins
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


def density_forecast_param(yp, sigma, rankmatrix, error_std, dof=np.zeros(24)):
    """creates a density forecast for Yp with Schaake Schuffle and parametric margins
            Parameters
           ----------
           yp: numpy.array
                    24-dimensional array with point-predictions of day ahead prices
           sigma: numpy.array
                    Variance prediction for each hour

           rankmatrix: numpy.array
                    Matrix with rank positions of forecast samples
           error_std:  numpy.array
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
    errordist = error_std.copy()
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
           rankmatrix:  numpy.array
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


def _get_rugarch_model(p: int, q: int, arch_p: int, arch_q: int, dist: str):
    """ Function to get rugarch model with specified parameters

    Parameters
    ----------
    p: AR-order
    q: MA-order
    arch_p: GARCH-order
    arch_q: GARCH-order
    dist: ['norm', 'std']

    Returns
    -------
    model: R-ugarchspec
    rugarch: link to library(rugarch)
    """
    rugarch = importr('rugarch')
    variance_model = robjects.ListVector(
        {'model': "sGARCH",
         'garchOrder': robjects.IntVector([arch_p, arch_q])})
    mean_model = robjects.ListVector(
        {'armaOrder': robjects.IntVector([p, q]),
         'include.mean': True})
    # Params auskommentieren für freie Wahl dof und andere dist!!!
    if dist == 'std':
        fix_df = 3
        if fix_df != 0:
            params = robjects.ListVector({'shape': fix_df})
            model = rugarch.ugarchspec(variance_model=variance_model, mean_model=mean_model,
                                       distribution_model=dist, fixed_pars=params)
        else:
            model = rugarch.ugarchspec(variance_model=variance_model, mean_model=mean_model,
                                       distribution_model=dist)
    else:
        model = rugarch.ugarchspec(variance_model=variance_model, mean_model=mean_model,
                                   distribution_model=dist)

    return model, rugarch

# ------------------------------ Additional Functions for Experiments ------------------------------------#
def dm_test(vgl, v00):
    '''
    Perfoms Diebold-Mariano test between two forecasts evaluated with same scoring rule
    :param vgl: List of Scoring rule results1
    :param v00: List of Scoring rule results2
    :return: Test-statistic value: t_value, p-value
    '''
    n = len(vgl)
    vgl = np.array(vgl)
    v00 = np.array(v00)
    delta_i = vgl - v00
    delta = np.sum(delta_i) / n
    std_delta = np.sqrt(np.var(delta_i) / n)
    t_value = delta / std_delta
    if t_value > 0:
        p_value = 2 * (1 - st.norm.cdf(t_value))
    else:
        p_value = 2 * st.norm.cdf(t_value)
   
    return t_value, p_value


def update_matrix(newvalue, matrix):
    '''

    :param newvalue:    np-array (1xd) that should be added to the matrix (nxd)
    :param matrix:      np-array (nxd)
    :return: matrix     updated np array
    '''
    matrix = np.vstack((matrix, newvalue))
    matrix = matrix[1:, :]

    return matrix


def meancrps(x_preds, x_test):
    '''Calculates Energy score for ensemble forecast
    Input:
        x_preds: np array or dataframe (nxd) with ensemble members
        x_test: np array or dataframe (1xd) point to be assesed
    Output:
        score: float with VariogrammScore'''
    [n, d] = x_preds.shape
    crsum = 0
    for i in range(0, d):
        test = x_preds[:, i]
        point = x_test[i]
        cr = ps.crps_ensemble(point, test)
        crsum += cr
    crtot = crsum / d
    return crtot


def rank_histogram_plot(multranks, m, name):
    '''
    Plot histogram and return
    :param multranks: list with ranks
    :param m: int number of ranks
    :param name: str with name
    :return: deviation
    '''
    multranks = np.array(multranks)
    # reshape the multranks into a 2D array of shape (n,1) where n is the size of the array.
    multranks = np.reshape(multranks, newshape=(-1, 1))
    # create an array filled with zeroes of shape (1, m+1)
    ran_freq = np.zeros(shape=(1, m + 1))
    for i in range(0, m + 1):
        # count the number of non-zero elements in the array where the elements are equal to i+1 and assign to the ran_freq
        ran_freq[0, i] = np.count_nonzero(multranks == i + 1)
    binnumb = m + 1
    plt.hist(multranks / (m + 1), bins=19, color='lightgrey', edgecolor='black', linewidth=1.2)
    x_coordinates = [0, 1]
    leng = multranks.shape[0]
    y_coordinates = [leng / binnumb, leng / binnumb]
    plt.axis('off')
    plt.gca().spines["top"].set_alpha(0.0)
    plt.gca().spines["bottom"].set_alpha(0.5)
    plt.gca().spines["right"].set_alpha(0.0)
    plt.gca().spines["left"].set_alpha(0.5)
    # save the plot as an image
    plt.tight_layout()
    plt.savefig(name + '.png')
    plt.show()

    # compute the deviation of the multrank
    multrank_dev = ran_freq / np.sum(ran_freq) - (1 / (m + 1))
    multrank_dev = np.sum(np.absolute(multrank_dev))

    return multrank_dev


def energyscore(x_preds, x_test):
    '''Calculates Energy score for ensemble forecast
    Input:
        x_preds: np array or dataframe (nxd) with ensemble members
        x_test: np array or dataframe (1xd) point to be assesed
    Output:
        score: float with EnergyScore'''
    # change to array typ
    x_preds = np.array(x_preds)
    x_test = np.array(x_test)
    [n, d] = x_preds.shape
    if n == d:
        print("prediction data is given in form of a nxd 2d matrix")
    elif len(x_preds) == len(x_test):
        x_preds = np.transpose(x_preds)
        print("prediction data is transposed into a nxd 2d matrix")
    sum1 = 0
    sum2 = 0
    for i in range(n):
        sum1 += LA.norm(x_preds[i] - x_test)
    for i in range(n):
        for j in range(n):
            sum2 += LA.norm(x_preds[i] - x_preds[j, :])

    score = 1 / n * sum1 - 1 / (2 * n ** 2) * sum2
    return score


def copula_matrix_u(data, name):
    '''
    Plot copula matrix
    :param data: np-array (nx24) of u-values
    :param name: str with name
    :return: -
    '''
    ncol = 24
    fig, axs = plt.subplots(ncol, ncol, figsize=(20, 20))
    for i in range(0, ncol):
        for j in range(0, ncol):
            if i == j:
                axs[i, j].hist(data[:, j], bins=10, density=True, color='tab:red')
                axs[i, j].set_xlim(0, 1)
                plt.setp(axs[i, j].get_xticklabels(), visible=False)
                plt.setp(axs[i, j].get_yticklabels(), visible=False)
            else:
                axs[i, j].scatter(data[:, i], data[:, j], s=0.1, color='tab:red')
                plt.setp(axs[i, j].get_xticklabels(), visible=False)
                plt.setp(axs[i, j].get_yticklabels(), visible=False)
                axs[i, j].set_xlim(0, 1)
                axs[i, j].set_ylim(0, 1)
    scattername = name
    # Uncomment to save results as pdf or png.
    # plt.savefig(scattername + '.pdf',format='pdf')
    plt.savefig(scattername + '.png',dpi=250)

    plt.show()


def average_rank(ens, test):
    '''
    Calulate the average rank
    :param ens: np-array of forecast distribution
    :param test: np-array with realized value
    :return: np-array with multivariate rank
    '''
    ens = np.array(ens)
    m = np.size(ens, axis=0)
    test = np.array(test)
    dim = np.size(ens, axis=1)

    new = np.vstack((test, ens))
    k = 0

    rankmatrix = st.rankdata(new, axis=0)
    avrank = np.mean(rankmatrix, axis=1)

    helper_low = np.count_nonzero(avrank < avrank[0])
    helper_high = np.count_nonzero(avrank <= avrank[0])
    multranks = np.random.randint(helper_low + 1, helper_high + 1, size=1, dtype='int')

    return multranks


def filter_df_by_h0_notna(df):
    """
    This function filters the input dataframe by the 'h0' column, removes any rows with 'null' values, and returns the filtered dataframe.

    Parameters:
    df (pandas.DataFrame): The dataframe to filter.

    Returns:
    pandas.DataFrame: The filtered dataframe with only rows where 'h0' is not 'null'.
    """
    df = df[df.h0.notna()]  # filter by h0 column
    nzero = np.size(df, axis=0) - df.isnull().sum()[0]  # count of the remaining rows
    if nzero != 0:
        df = df.iloc[:nzero, :]  # remove any rows with null values
    return df


def round_and_stringify(val):
    '''
    Rounds and stringies values for printing
    :param val: values
    :return: rounded, stringified value
    '''
    # Use the round function to round the input value to 3 decimal places
    rounded_val = round(val, 3)

    # Check if the rounded value is 0
    if rounded_val == 0:
        # If the rounded value is 0, return the string "$<$0.001"
        return "$<$0.001"
    # Check if the rounded value is 1
    elif rounded_val == 1:
        # If the rounded value is 1, return the string "$>$0.999"
        return "$>$0.999"
    else:
        # If the rounded value is neither 0 nor 1, convert it to a string and return it
        return str(rounded_val)

# ------------------------------ Additional Functions for ENTSO-E Forecast ------------------------------------#
def learn_multivariate_density_ENTSOE(yp, error, timeseries_treatment=True, lenght_error_learning=90,
                               length_dependence_learning=90,
                               param_margin=False, param_dependence=False, ts_dist='norm'):
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
           param_margin: boolean
                    use parametric margins if True
           param_dependence: boolean
                    use parametric Gaussian copula if True
           ts_dist: str
                    Specify distribution for parametric margin. 'norm' for Gaussian, 'std' for Students t-distribution
           Returns
           -------
           dens:   numpy.array
                    """
    # Sanity checks
    if np.size(error, axis=0) < lenght_error_learning:
        raise ValueError('Given error series is shorter then specified number of errors for error learning!')

    if lenght_error_learning < length_dependence_learning:
        raise ValueError('Lenght_error_learning must be at least of same size then length_dependence_learning!')

    bias = np.zeros(24)
    dof = np.zeros(24)
    sigma = np.ones(24)
    resid = np.zeros(shape=error.shape)

    # Time Series Specifications
    if timeseries_treatment == True:
        length_ts_fitting = 365
        if length_ts_fitting > np.size(error, axis=0):
            warnings.warn("No full year of historic errors is used for time series fitting!")
            length_ts_fitting = np.size(error, axis=0)
        if np.size(error, axis=0) < 90:
            raise ValueError('Not enough historic errors for time series standardization!')

        for a in range(24):
            errors = error[:, a]
            my_order = (1, 0, 0)
            my_seasonal_order = (1, 0, 0, 7)
            model = SARIMAX(errors, order=my_order, seasonal_order=my_seasonal_order,
                            initialization='approximate_diffuse')
            model_fit = model.fit(disp=False)
            bias[a] = model_fit.forecast()
            resid[:, a] = model_fit.resid

        error_std = resid[-lenght_error_learning:, :]

    # Non-standardized Errors (Raw Error)
    else:
        error_std = error[-lenght_error_learning:, :]

    # Dependence Learning
    data_dependence = error_std[-length_dependence_learning:, :].copy()
    rankmatrix = get_rankmatrix(data_dependence, param_dependence)

    # Forecast
    if param_margin == False:
        density = density_forecast_nonparam(yp - bias, sigma ** 2, rankmatrix, error_std)
    if param_margin == True:
        density = density_forecast_param(yp - bias, sigma ** 2, rankmatrix, error_std, dof)

    return density, sigma, bias, dof
