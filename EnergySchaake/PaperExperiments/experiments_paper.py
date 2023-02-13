##
import pandas as pd
import numpy as np
import argparse
import os
from functions_energy_schaake import *
import scipy.stats as st
from epftoolbox.data import read_data
from epftoolbox.models import LEAR
import warnings

warnings.filterwarnings("ignore")


# ------------------------------ EXTERNAL PARAMETERS ------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=market,
                    help='Market under study. If it not one of the standard ones, the file name' +
                         'has to be provided, where the file has to be a csv file')


parser.add_argument("--years_test", type=int, default=3,
                    help='Number of years (a year is 364 days) in the test dataset. ')

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
path_recalibration_folder = os.path.join('.', 'experimental_files')

model = LEAR(calibration_window=calibration_window)
named = str(dataset)

# ------------------------------ DEFINING NEEDED ARRAYS ------------------------------------#
# Print setting
print('Dataset: ' + dataset + ' | length error learning: ' + str(
    error_marginals) + ' | length error_learning_timeseries = ' + str(error_learning_timeseries))

# Defining train and testing data
df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                              begin_test_date=None, end_test_date=None)

# Defining empty forecast array and the real values to be predicted in a more friendly format
forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])

# Defining arrays for learning margins
forecast_learn_out = forecast.iloc[:error_learning_timeseries, :].copy()
forecast_density = forecast.iloc[error_learning_timeseries:, :].copy()

# Extract real values for testing and error calculation
real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

# Define empty arrays to store results
u = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
u_00_ts = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
u_11_ts = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
u_00_raw = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
forecast_learn_out_dates = forecast_learn_out.index
forecast_density_dates = forecast_density.index

# Define dictonaries for evaluation data
density_forecast_dics = {
    "vgl_00_ts": {},
    "vgl_11_ts": {},
    "00_ts": {},
    "11_ts": {},
    "vgl_00_raw": {},
    "00_raw": {}
}

es = {
    "vgl_00_ts": [],
    "vgl_11_ts": [],
    "00_ts": [],
    "11_ts": [],
    "vgl_00_raw": [],
    "00_raw": []
}

crps = {
    "11_ts": [],
    "00_ts": [],
    "00_raw": []
}

avg_ranks = {
    "vgl_00_ts": [],
    "vgl_11_ts": [],
    "00_ts": [],
    "11_ts": [],
    "vgl_00_raw": [],
    "00_raw": []
}

# ------------------------------ GET FORECASTING ERRORS OF SPECIFIED TIMEFRAMES ------------------------------------#
# Loop over dates for time series fitting
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

forecast_learn_out = forecast_learn_out.to_numpy()

# ------------------------------ FORECASTING & EVALUATION ------------------------------------#
# Counter
n = 0
# Loop over Forecasting days
for date in forecast_density_dates:
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
                                                                          param_margin=False, param_dependence=False,ts_dist='norm')

    forecast_distribution_11_ts, _, _, _ = learn_multivariate_density(Yp, forecast_learn_out,
                                                                          timeseries_treatment=True,
                                                                          lenght_error_learning=error_marginals,
                                                                          length_dependence_learning=dep_learning,
                                                                          param_margin=True, param_dependence=True,ts_dist='norm')

    forecast_distribution_00_raw, sigma_raw, bias_raw, _ = learn_multivariate_density(Yp, forecast_learn_out,
                                                                           timeseries_treatment=False,
                                                                           lenght_error_learning=error_marginals,
                                                                           length_dependence_learning=dep_learning,
                                                                           param_margin=False, param_dependence=False,ts_dist='norm')

    # Forecasts ignoring the copula
    forecast_distribution_vgl_00_ts,  _, _, _ = learn_multivariate_density(Yp, forecast_learn_out,
                                                                              timeseries_treatment=True,
                                                                              lenght_error_learning=error_marginals,
                                                                              length_dependence_learning=dep_learning,
                                                                              param_margin=False,
                                                                              param_dependence=False,ts_dist='norm')
    forecast_distribution_vgl_00_ts = np.apply_along_axis(np.random.permutation, 0, forecast_distribution_vgl_00_ts)

    forecast_distribution_vgl_11_ts, _, _, _ = learn_multivariate_density(Yp, forecast_learn_out,
                                                                              timeseries_treatment=True,
                                                                              lenght_error_learning=error_marginals,
                                                                              length_dependence_learning=dep_learning,
                                                                              param_margin=True, param_dependence=True,ts_dist='norm')
    forecast_distribution_vgl_11_ts = np.apply_along_axis(np.random.permutation, 0, forecast_distribution_vgl_11_ts)

    forecast_distribution_vgl_00_raw,  _, _, _ = learn_multivariate_density(Yp, forecast_learn_out,
                                                                               timeseries_treatment=False,
                                                                               lenght_error_learning=error_marginals,
                                                                               length_dependence_learning=dep_learning,
                                                                               param_margin=False,
                                                                               param_dependence=False,ts_dist='norm')
    forecast_distribution_vgl_00_raw = np.apply_along_axis(np.random.permutation, 0, forecast_distribution_vgl_00_raw)

    # Save Forecasts and Evaluate
    c = real_values.loc[date].values
    density_forecast_dics['vgl_00_ts'][date] = forecast_distribution_vgl_00_ts
    density_forecast_dics['vgl_11_ts'][date] = forecast_distribution_vgl_11_ts
    density_forecast_dics['00_ts'][date] = forecast_distribution_00_ts
    density_forecast_dics['11_ts'][date] = forecast_distribution_11_ts
    density_forecast_dics['vgl_00_raw'][date] = forecast_distribution_vgl_00_raw
    density_forecast_dics['00_raw'][date] = forecast_distribution_00_raw

    es_00_vgl_ts = energyscore(forecast_distribution_vgl_00_ts, c)
    es_11_vgl_ts = energyscore(forecast_distribution_vgl_11_ts, c)
    es_00_ts = energyscore(forecast_distribution_00_ts, c)
    es_11_ts = energyscore(forecast_distribution_11_ts, c)
    es_00_vgl_raw = energyscore(forecast_distribution_vgl_00_raw, c)
    es_00_raw = energyscore(forecast_distribution_00_raw, c)

    crps_11_ts = meancrps(forecast_distribution_11_ts, Y_echt)
    crps_00_ts = meancrps(forecast_distribution_00_ts, Y_echt)
    crps_00_raw = meancrps(forecast_distribution_00_raw, Y_echt)

    print(' - {} - '.format(str(date)[:10]))
    es['vgl_00_ts'].append(es_00_vgl_ts)
    es['vgl_11_ts'].append(es_11_vgl_ts)
    es['00_ts'].append(es_00_ts)
    es['11_ts'].append(es_11_ts)
    es['vgl_00_raw'].append(es_00_vgl_raw)
    es['00_raw'].append(es_00_raw)

    crps['11_ts'].append(crps_11_ts)
    crps['00_ts'].append(crps_00_ts)
    crps['00_raw'].append(crps_00_raw)

    avg_ranks['vgl_00_ts'].append(average_rank(forecast_distribution_vgl_00_ts, Y_echt))
    avg_ranks['vgl_11_ts'].append(average_rank(forecast_distribution_vgl_11_ts, Y_echt))
    avg_ranks['00_ts'].append(average_rank(forecast_distribution_00_ts, Y_echt))
    avg_ranks['11_ts'].append(average_rank(forecast_distribution_11_ts, Y_echt))
    avg_ranks['vgl_00_raw'].append(average_rank(forecast_distribution_vgl_00_raw, Y_echt))
    avg_ranks['00_raw'].append(average_rank(forecast_distribution_00_raw, Y_echt))

    # Get latest raw error
    Y_help_raw = (Yp - Y_echt)
    # Update array with historic errors
    forecast_learn_out = update_matrix(Y_help_raw, forecast_learn_out)
    # Get realized, standardized errors for parametric PIT-Plots
    Y_help_ts = (Yp - bias_ts - Y_echt) / sigma_ts

    # Calculate u-values for PIT-Matrix non-parametrically
    rank_ts = np.zeros(shape=(error_marginals + 1, 24))
    rank_ts[0] = Y_echt
    rank_ts[1:] = forecast_distribution_00_ts
    ranks_ts = st.rankdata(rank_ts, axis=0) / (rank_ts.shape[0] + 1)
    u_00_ts.iloc[u_00_ts.index.get_loc(date), :] = ranks_ts[0, :]

    rank_raw = np.zeros(shape=(error_marginals + 1, 24))
    rank_raw[0] = Y_echt
    rank_raw[1:] = forecast_distribution_00_raw
    ranks_raw = st.rankdata(rank_raw, axis=0) / (rank_raw.shape[0] + 1)
    u_00_raw.iloc[u_00_raw.index.get_loc(date), :] = ranks_raw[0, :]

    # Get parametric u value
    if (dof_ts == 0).all():
        # normal distribution
        u_11_ts.iloc[u_11_ts.index.get_loc(date), :] = st.norm.cdf(Y_help_ts, loc=0, scale=1)
    else:
        # t-distribution
        u_11_ts.iloc[u_11_ts.index.get_loc(date), :] = st.t.cdf(Y_help_ts, loc=0, scale=1, df=dof_ts)

    # Print daily result
    print("ES: " + str(sum(es['00_ts'])) + '   |   ' + str(sum(es['11_ts'])) + '   |   ' + str(
        sum(es['00_raw'])) + '   |   ' + str(
        sum(es['vgl_00_ts'])) + '   |   ' + str(sum(es['vgl_11_ts'])) + '   |   ' + str(
        sum(es['vgl_00_raw'])))
    print("CRPS: " + str(sum(crps['00_ts'])) + '   |   ' + str(sum(crps['11_ts'])) + '   |   ' + str(
        sum(crps['00_raw'])))
    # Add 1 to counter
    n += 1

# ------------------------------ PLOT AND PRINT RESULTS ------------------------------------#
## Plot PIT-Matrix
u_00_numpy_ts = filter_df_by_h0_notna(u_00_ts)
copula_matrix_u(u_00_numpy_ts.to_numpy(), 'non11_ts' + named)

u_11_numpy_ts = filter_df_by_h0_notna(u_11_ts)
copula_matrix_u(u_11_numpy_ts.to_numpy(), '11' + named)

u_00_numpy_raw = filter_df_by_h0_notna(u_00_raw)
copula_matrix_u(u_00_numpy_raw.to_numpy(), 'raw' + named)

# Plot average rank histograms
_ = rank_histogram_plot(avg_ranks['vgl_00_ts'], error_marginals, 'vgl_00_ts_av_' + named)
_ = rank_histogram_plot(avg_ranks['vgl_11_ts'], error_marginals, 'vgl_11_ts_av_' + named)
_ = rank_histogram_plot(avg_ranks['00_ts'], error_marginals, '00_ts_av_' + named)
_ = rank_histogram_plot(avg_ranks['11_ts'], error_marginals, '11_ts_av_' + named)
_ = rank_histogram_plot(avg_ranks['vgl_00_raw'], error_marginals, 'vgl_00_raw_av_' + named)
_ = rank_histogram_plot(avg_ranks['00_raw'], error_marginals, '00_raw_av_' + named)

# Final output
if n == 0:
    print('No data to validate!')
else:
    res_es_00_vgl_ts = sum(es['vgl_00_ts']) / n
    res_es_11_vgl_ts = sum(es['vgl_11_ts']) / n
    res_es_00_ts = sum(es['00_ts']) / n
    res_es_11_ts = sum(es['11_ts']) / n
    res_es_00_vgl_raw = sum(es['vgl_00_raw']) / n
    res_es_00_raw = sum(es['00_raw']) / n

    crps_11_res_ts = sum(crps['11_ts']) / n
    crps_00_res_ts = sum(crps['00_ts']) / n
    crps_00_res_raw = sum(crps['00_raw']) / n

    _, p_vgl_00_es_ts = dm_test(es['00_ts'], es['vgl_00_ts'])
    _, p_vgl_11_es_ts = dm_test(es['00_ts'], es['vgl_11_ts'])
    _, p_11_es_ts = dm_test(es['00_ts'], es['11_ts'])
    _, p_vgl_00_es_raw = dm_test(es['00_ts'], es['vgl_00_raw'])
    _, p_00_es_raw = dm_test(es['00_ts'], es['00_raw'])
    _, p_vgl_11_crps_ts = dm_test(crps['00_ts'], crps['11_ts'])
    _, p_vgl_00_crps_raw = dm_test(crps['00_ts'], crps['00_raw'])

    p_vgl_00_es_ts = round_and_stringify(p_vgl_00_es_ts)
    p_vgl_11_es_ts = round_and_stringify(p_vgl_11_es_ts)
    p_11_es_ts = round_and_stringify(p_11_es_ts)
    p_vgl_00_es_raw = round_and_stringify(p_vgl_00_es_raw)
    p_00_es_raw = round_and_stringify(p_00_es_raw)
    p_vgl_11_crps_ts = round_and_stringify(p_vgl_11_crps_ts)
    p_vgl_00_crps_raw = round_and_stringify(p_vgl_00_crps_raw)

    # Printing result
    print('Dataset: ' + dataset + ' | length error learning: ' + str(
        error_marginals) + ' | length error_learning_timeseries = ' + str(
        error_learning_timeseries))
    print('ES  CRPS')

    print('Schaake-NP  &  ' + str(round(res_es_00_ts, 3)) + '&  ' + str(
        round(crps_00_res_ts, 3)) + '&1.000 & 1.000' + r"\\")
    print('Schaake-P  &  ' + str(round(res_es_11_ts, 3)) + '&  ' + str(
        round(crps_11_res_ts, 3)) + '& ' + str(p_11_es_ts) + ' &' + str(p_vgl_11_crps_ts) + r"\\")
    print('Schaake-Raw  &  ' + str(round(res_es_00_raw, 3)) + '& ' + str(
        round(crps_00_res_raw, 3)) + '& ' + str(p_00_es_raw) + ' &' + str(p_vgl_00_crps_raw) + r"\\")
    print('I-NP  &  ' + str(round(res_es_00_vgl_ts, 3)) + '& ' + str(
        round(crps_00_res_ts, 3)) + '& ' + str(p_vgl_00_es_ts) + '& 1.000' + r"\\")
    print('I-P  &  ' + str(round(res_es_11_vgl_ts, 3)) + '& ' + str(
        round(crps_11_res_ts, 3)) + '& ' + str(p_vgl_11_es_ts) + ' &' + str(p_vgl_11_crps_ts) + r"\\")
    print('I-Raw  &  ' + str(round(res_es_00_vgl_raw, 3)) + '& ' + str(
        round(crps_00_res_raw, 3)) + '& ' + str(p_vgl_00_es_raw) + '& ' + str(p_vgl_00_crps_raw) + r"\\")

print('Finished calculatios')


