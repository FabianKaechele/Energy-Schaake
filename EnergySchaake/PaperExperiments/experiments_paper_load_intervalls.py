##
import pandas as pd
import numpy as np
import argparse
import os
import datetime
from epftoolbox.data import read_data
from epftoolbox.models import LEAR
import warnings
from functions_energy_shaake import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import asgl
warnings.filterwarnings("ignore")


def main():
    # ------------------------------ EXTERNAL 11ETERS ------------------------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='DE',
                        help='Market under study. If it not one of the standard ones, the file name' +
                             'has to be provided, where the file has to be a csv file')

    ##########Hier mehr dazu!
    parser.add_argument("--years_test", type=int, default=3,
                        help='Number of years (a year is 364 days) in the test dataset. Used if ' +
                             ' begin_test_date and end_test_date are not provided.')
    ##########Hier mehr dazu!
    parser.add_argument("--calibration_window", type=int, default=2 * 364,
                        help='Number of days used in the training dataset for recalibration')

    parser.add_argument("--error_marginals", type=int, default=90,
                        help='Number of days used for constructing margins in forecast - usually same length as dependence learning ')

    parser.add_argument("--dep_learning", type=int, default=90,
                        help='Number of days in the Dependence Learning Phase')

    parser.add_argument("--error_learning_timeseries", type=int, default=364,
                        help='Number of days in the Error Learning Phase for time series fitting on margins')

    args = parser.parse_args()
    dataset = args.dataset
    years_test = args.years_test
    calibration_window = args.calibration_window
    error_marginals = args.error_marginals
    dep_learning = args.dep_learning
    error_learning_timeseries = args.error_learning_timeseries
    path_datasets_folder = os.path.join('.', 'datasets')
    path_datasets_folder = os.path.join(os.getcwd(), 'datasets')


    # Defining train and testing data
    df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                                  begin_test_date=None, end_test_date=None)

    # Defining empty forecast array and the real values to be predicted in a more friendly format
    forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
    forecast_learn_out = forecast.iloc[:error_learning_timeseries, :].copy()
    evaluation_data = forecast.iloc[error_learning_timeseries:, :].copy()

    forecast_learn_out_dates = forecast_learn_out.index
    evaluation_dates = evaluation_data.index
    model = LEAR(calibration_window=calibration_window)

    # hourly weights mean over all days and within hour
    weights_day_kw = np.array([68.59,
                               62.46,
                               56.30,
                               54.83,
                               58.73,
                               62.70,
                               67.84,
                               91.68,
                               131.29,
                               152.31,
                               158.47,
                               164.21,
                               158.81,
                               141.11,
                               121.96,
                               113.49,
                               109.34,
                               113.24,
                               106.59,
                               92.96,
                               89.51,
                               84.33,
                               76.84,
                               70.21
                               ])
    weights_day_mw = weights_day_kw / 1000

    # ------------------------------ GET FORECASTING ERRORS OF SPECIFIED TIMEFRAMES ------------------------------------#
    # Loop over dates for time series fitting
    forecast_easy = []
    for date in forecast_learn_out_dates:
        data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

        # Safe real values
        Y_echt = np.copy(data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'])

        # Set the real prices for current date to NaN
        data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

        # Recalibrating the model with the most up-to-date available data and making a prediction
        # for the next day
        Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date,
                                                     calibration_window=calibration_window)
        # Safe forecast and calculate error
        forecast.loc[date, :] = Yp
        forecast_learn_out.loc[date, :] = Yp - Y_echt
        print('{} - Out of sample forecast error saved'.format(str(date)[:10]))
        forecast_easy.append(np.sum(weights_day_mw * Y_echt))

    forecast_learn_out = forecast_learn_out.to_numpy()

    # Define some helpers
    n = 0
    max_echt = -100000
    min_echt = 100000
    vars_outKI = {'vgl_00': 0, '00': 0, 'vgl_raw': 0, '00_raw': 0, 'easy': 0, 'qr': 0, 'timeseries_easy': 0, 'kde': 0}
    widths = {'00': 0, 'vgl': 0, 'vgl_raw': 0, '00_raw': 0, 'easy': 0, 'kde': 0, 'qr': 0, 'timeseries_easy': 0}
    qr_low_ls = []
    qr_up_ls = []
    forecast_easy = np.array(forecast_easy)[-error_marginals:]
    forecast_easy_ranked = np.sort(forecast_easy, axis=None)
    forecast_intervalls_vgl = np.zeros(shape=(dep_learning, evaluation_data.shape[0]))
    forecast_intervalls_00 = np.zeros(shape=(dep_learning, evaluation_data.shape[0]))
    forecast_intervalls_vgl_raw = np.zeros(shape=(dep_learning, evaluation_data.shape[0]))
    forecast_intervalls_00_raw = np.zeros(shape=(dep_learning, evaluation_data.shape[0]))
    cost_echt_array = np.zeros(evaluation_data.shape[0])

    # Start evaluation period
    for date in evaluation_dates:

        # For simulation purposes, we assume that the available data is
        # the data up to current date where the prices of current date are not known
        data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

        # Safe real values
        Y_echt = np.copy(data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'])

        # We set the real prices for current date to NaN in the dataframe of available data
        data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

        # Recalibrating the model with the most up-to-date available data and making a prediction
        # for the next day
        Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date,
                                                     calibration_window=calibration_window)


        # Forecast all versions
        forecast_distribution_00_ts, sigma_ts, bias_ts, dof_ts = learn_multivariate_density(Yp, forecast_learn_out,
                                                                                            timeseries_treatment=True,
                                                                                            lenght_error_learning=error_marginals,
                                                                                            length_dependence_learning=dep_learning,
                                                                                            param_margin=False,
                                                                                            param_dependence=False,
                                                                                            ts_dist='norm')
        forecast_distribution_vgl_00_ts = np.apply_along_axis(np.random.permutation, 0, forecast_distribution_00_ts)

        forecast_distribution_00_raw, _, _, _ = learn_multivariate_density(Yp, forecast_learn_out,
                                                                           timeseries_treatment=False,
                                                                           lenght_error_learning=error_marginals,
                                                                           length_dependence_learning=dep_learning,
                                                                           param_margin=False,
                                                                           param_dependence=False,
                                                                           ts_dist='norm')
        forecast_distribution_vgl_00_raw = np.apply_along_axis(np.random.permutation, 0, forecast_distribution_00_raw)

        # QR
        idx = forecast.index
        helper = int(forecast.index.get_indexer([date]))
        data_qr = forecast.iloc[helper - error_marginals:helper, :]
        qr_low_model = asgl.ASGL('qr','lasso',lambda1=0.1, tau= 0.033)
        qr_low_model.fit(y=forecast_easy, x=data_qr)
        qr_low = qr_low_model.predict(x_new=Yp)[0][0]
        qr_low_ls.append(qr_low)

        qr_up_model = asgl.ASGL('qr', 'lasso', lambda1=0.1, tau=0.966)
        qr_up_model.fit(y=forecast_easy, x=data_qr)
        qr_up = qr_up_model.predict(x_new=Yp)[0][0]
        qr_up_ls.append(qr_up)


        # KDE
        dens = sm.nonparametric.KDEUnivariate(forecast_easy)
        dens.fit(bw='silverman')
        kde_up = st.mstats.mquantiles(dens.endog, np.linspace(0, 1, 90))[87]  # dens.icdf[87]
        kde_low = st.mstats.mquantiles(dens.endog, np.linspace(0, 1, 90))[2]  # dens.icdf[2]

        # AR-GARCH
        model_archgarchr_total = r_garch_forecast_t(forecast_easy, p=1, q=0, arch_p=1,
                                                    arch_q=1)
        mu_total = model_archgarchr_total['Mu']
        sigma_total = model_archgarchr_total['Sigma']
        timeseries_easy_up = st.norm.ppf(0.966, mu_total, sigma_total)
        timeseries_easy_low = st.norm.ppf(0.033, mu_total, sigma_total)

        # Cost Quantiles Schaake Schuffle TS
        cost_day_vgl = np.sum(weights_day_mw * forecast_distribution_vgl_00_ts, axis=1)
        cost_day_00 = np.sum(weights_day_mw * forecast_distribution_00_ts, axis=1)
        vgl_sorted = np.sort(cost_day_vgl, axis=0)
        sorted_00 = np.sort(cost_day_00, axis=0)
        forecast_intervalls_vgl[:, n] = vgl_sorted
        forecast_intervalls_00[:, n] = sorted_00

        # Cost Quantiles Schaake Schuffle Raw
        cost_day_vgl_raw = np.sum(weights_day_mw * forecast_distribution_vgl_00_raw, axis=1)
        cost_day_00_raw = np.sum(weights_day_mw * forecast_distribution_00_raw, axis=1)
        vgl_sorted_raw = np.sort(cost_day_vgl_raw, axis=0)
        sorted_00_raw = np.sort(cost_day_00_raw, axis=0)
        forecast_intervalls_vgl_raw[:, n] = vgl_sorted_raw
        forecast_intervalls_00_raw[:, n] = sorted_00_raw

        # Real cost of profile
        cost_echt = np.sum(weights_day_mw * Y_echt)
        cost_echt = np.sum(cost_echt)
        cost_echt_array[n] = cost_echt

        out_00 = 0
        out_vgl = 0
        max_echt = max(cost_echt, max_echt)
        min_echt = min(cost_echt, min_echt)

        # Check if real value is in interval
        if cost_echt <= vgl_sorted[2] or cost_echt >= vgl_sorted[87]:
            vars_outKI['vgl_00'] += 1
            out_vgl = 1

        if cost_echt <= sorted_00[2] or cost_echt >= sorted_00[87]:
            vars_outKI['00'] += 1
            out_00 = 1

        if cost_echt <= vgl_sorted_raw[2] or cost_echt >= vgl_sorted_raw[87]:
            vars_outKI['vgl_raw'] += 1
            out_vgl = 1

        if cost_echt <= sorted_00_raw[2] or cost_echt >= sorted_00_raw[87]:
            vars_outKI['00_raw'] += 1
            out_00 = 1

        if cost_echt <= forecast_easy_ranked[2] or cost_echt >= forecast_easy_ranked[87]:
            vars_outKI['easy'] += 1

        if cost_echt <= qr_low or cost_echt >= qr_up:
            vars_outKI['qr'] += 1

        if cost_echt <= kde_low or cost_echt >= kde_up:
            vars_outKI['kde'] += 1

        if cost_echt <= timeseries_easy_low or cost_echt >= timeseries_easy_up:
            vars_outKI['timeseries_easy'] += 1

        # PINAW calculations
        widths['00'] += sorted_00[87] - sorted_00[2]
        widths['vgl_00'] += vgl_sorted[87] - vgl_sorted[2]
        widths['00_raw'] += sorted_00_raw[87] - sorted_00_raw[2]
        widths['vgl_raw'] += vgl_sorted_raw[87] - vgl_sorted_raw[2]
        widths['easy'] += forecast_easy_ranked[87] - forecast_easy_ranked[2]
        widths['qr'] += qr_up - qr_low
        widths['kde'] += kde_up - kde_low
        widths['timeseries_easy'] += timeseries_easy_up - timeseries_easy_low

        # Update values
        forecast_learn_out = update_matrix(Yp - Y_echt, forecast_learn_out)
        forecast.loc[date, :] = Yp
        forecast_easy = np.hstack((forecast_easy[1:], cost_echt))
        forecast_easy_ranked = np.sort(forecast_easy, axis=None)

        # Print
        n += 1
        print('{} KI tested'.format(str(date)[:10]))
        print('I-NP of KI: ' + str(out_vgl) + ' | Schaake-NP out of KI: ' + str(
            out_00) + ' | I-NP out of KItotal: ' + str(
            vars_outKI['vgl_00'] / n) + ' | Schaake-NP out of KI total: ' + str(
            vars_outKI['00'] / n) + ' | VGL zu Univariate: ' + str(
            vars_outKI['easy'] / n) + ' | QR: ' + str(vars_outKI['qr'] / n) + ' | KDE: ' + str(
            vars_outKI['kde'] / n) + ' | TS_easy: ' + str(
            vars_outKI['timeseries_easy'] / n) + ' | I-Raw out of KItotal: ' + str(
            vars_outKI['vgl_raw'] / n) + ' | Schaake-Raw out of KI total: ' + str(vars_outKI['00_raw'] / n))
        print('FINAW I-NP: ' + str(widths['vgl_00'] / (n * (max_echt - min_echt))) + '| FINAW Schaake-NP: ' + str(
            widths['00'] / (n * (max_echt - min_echt))) + '| FINAW univariate: ' + str(
            widths['easy'] / (n * (max_echt - min_echt))) + '| FINAW QR: ' + str(
            widths['qr'] / (n * (max_echt - min_echt))) + '| FINAW KDE: ' + str(
            widths['kde'] / (n * (max_echt - min_echt))) + '| FINAW TS_easy: ' + str(
            widths['timeseries_easy'] / (n * (max_echt - min_echt))) + 'FINAW I-Raw: ' + str(
            widths['vgl_raw'] / (n * (max_echt - min_echt))) + '| FINAW Schaake-Raw: ' + str(
            widths['00_raw'] / (n * (max_echt - min_echt))))


if __name__ == "__main__":
    main()
