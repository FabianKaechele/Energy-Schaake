
import pandas as pd
import numpy as np
from functions_energy_schaake import *
import warnings
warnings.filterwarnings("ignore")

# Set Parameters
error_marginals = 90
dep_learning = 90
error_learning_timeseries = 364
dataset = 'entso_e'
index = 0

#  Load data
data = pd.read_excel('Load_DayAhead_Actual_DEU_2016to2020.xlsx')
time = data['Time']
load = data['Load']
load = np.array(load).reshape((-1, 24))
load = load[-730 - error_learning_timeseries - 2:, :]

forecast = data['Loadforecast']
forecast = np.array(forecast).reshape((-1, 24))
forecast = forecast[-730 - error_learning_timeseries - 2:, :]

# Crate empty arrays
forecast_std_error = np.empty(shape=forecast.shape)
forecast_learn_out = np.empty((error_learning_timeseries, 24))
forecast_learn_dist = np.empty((error_marginals, 24))
u_nonparam_ts = np.empty(shape=forecast.shape)
u_nonparam_raw = np.empty(shape=forecast.shape)
u_param = np.empty(shape=forecast.shape)

# Get error Time Series
error = forecast - load
for i in range(0, error_learning_timeseries):
    # Safe real values
    Y_echt = load[index, :]

    Yp = forecast[index, :]
    forecast_learn_out[i, :] = Yp - Y_echt
    print('{} - Out of sample forecast error saved ' + str(index+1))
    index += 1

density_forecast_dics = {
    "vgl_00_ts": {},
    "00_ts": {},
    "vgl_00_raw": {},
    "00_raw": {}
}

es = {
    "vgl_00_ts": [],
    "00_ts": [],
    "vgl_00_raw": [],
    "00_raw": []
}

crps = {
    "11_ts": [],
    "00_ts": [],
    "00_raw": []
}


# Forecast and Evaluate
n = 0
for date in range(0, 730):

    Y_echt = load[index, :]
    Yp = forecast[index, :]
    Yp = np.reshape(Yp, (1, 24))

    # Forecast
    forecast_distribution_00_ts,   _, _, _ = learn_multivariate_density_ENTSOE(Yp, forecast_learn_out,
                                                                          timeseries_treatment=True,
                                                                          lenght_error_learning=error_marginals,
                                                                          length_dependence_learning=dep_learning,
                                                                          param_margin=False, param_dependence=False)


    forecast_distribution_00_raw,   _, _, _ = learn_multivariate_density_ENTSOE(Yp, forecast_learn_out,
                                                                           timeseries_treatment=False,
                                                                           lenght_error_learning=error_marginals,
                                                                           length_dependence_learning=dep_learning,
                                                                           param_margin=False, param_dependence=False)

    # Forecasts ignoring the copula
    forecast_distribution_vgl_00_ts,   _, _, _ = learn_multivariate_density_ENTSOE(Yp, forecast_learn_out,
                                                                              timeseries_treatment=True,
                                                                              lenght_error_learning=error_marginals,
                                                                              length_dependence_learning=dep_learning,
                                                                              param_margin=False,
                                                                              param_dependence=False)
    forecast_distribution_vgl_00_ts = np.apply_along_axis(np.random.permutation, 0, forecast_distribution_vgl_00_ts)

    forecast_distribution_vgl_00_raw,   _, _, _ = learn_multivariate_density_ENTSOE(Yp, forecast_learn_out,
                                                                               timeseries_treatment=False,
                                                                               lenght_error_learning=error_marginals,
                                                                               length_dependence_learning=dep_learning,
                                                                               param_margin=False,
                                                                               param_dependence=False)
    forecast_distribution_vgl_00_raw = np.apply_along_axis(np.random.permutation, 0, forecast_distribution_vgl_00_raw)

    # Safe and evaluate
    density_forecast_dics['vgl_00_ts'][date] = forecast_distribution_vgl_00_ts
    density_forecast_dics['vgl_00_raw'][date] = forecast_distribution_vgl_00_raw
    density_forecast_dics['00_ts'][date] = forecast_distribution_00_ts
    density_forecast_dics['00_raw'][date] = forecast_distribution_00_raw

    c = Y_echt
    es_00_vgl = energyscore(forecast_distribution_vgl_00_ts, c)
    es_00_vgl_raw = energyscore(forecast_distribution_vgl_00_raw, c)
    es_00 = energyscore(forecast_distribution_00_ts, c)
    es_00_raw = energyscore(forecast_distribution_00_raw, c)


    crps_00 = meancrps(forecast_distribution_00_ts, Y_echt)
    crps_00_raw = meancrps(forecast_distribution_00_raw, Y_echt)

    print(
        '{} - saved'.format(str(index)))

    es['vgl_00_ts'].append(es_00_vgl)
    es['00_ts'].append(es_00)
    es['vgl_00_raw'].append(es_00_vgl_raw)
    es['00_raw'].append(es_00_raw)


    crps['00_ts'].append(crps_00)
    crps['00_raw'].append(crps_00_raw)
    n += 1
    # Saving the current prediction
    forecast_learn_out = update_matrix(Yp - Y_echt, forecast_learn_out)
    index += 1

# Nonzero rows
u_nonparam_numpy_ts = u_nonparam_ts[-728:, :]
u_nonparam_numpy_raw = u_nonparam_raw[-728:, :]

##
named = '0'
if n == 0:
    print('No data to validate!')
else:

    res_es_00_vgl = sum(es['vgl_00_ts']) / n
    res_es_00_vgl_raw = sum(es['vgl_00_raw']) / n
    res_es_00 = sum(es['00_ts']) / n
    res_es_00_raw = sum(es['00_raw']) / n


    crps_00_res = sum(crps['00_ts']) / n
    crps_00_res_raw = sum(crps['00_raw']) / n

    _, p_vgl_00_es = dm_test(es['00_ts'], es['vgl_00_ts'])
    _, p_vgl_00_es_raw = dm_test(es['00_ts'], es['vgl_00_raw'])
    _, p_es_00_raw = dm_test(es['00_ts'], es['00_raw'])
    _, p_vgl_crps = dm_test(crps['00_ts'], crps['00_raw'])

    p_vgl_00_es = round_and_stringify(p_vgl_00_es)
    p_vgl_00_es_raw = round_and_stringify(p_vgl_00_es_raw)
    p_es_00_raw = round_and_stringify(p_es_00_raw)
    p_vgl_crps = round_and_stringify(p_vgl_crps)

    ## printing result
    print('Dataset: ' + dataset )
    print('ES CRPS')

    print('Shaake-NP  &  ' + str(round(res_es_00, 3)) + '&  ' + '& ' + str(
        round(crps_00_res, 3)) + '&(- , -)' + r"\\")
    print('Shaake-Raw  &  ' + str(round(res_es_00_raw, 3)) + '&  ' + '& ' + str(
        round(crps_00_res_raw, 3)) + '& (' + str(round(p_es_00_raw, 3)) + ',' + str(
        round(p_vgl_crps, 3)) + ')' + r"\\")
    print('I-NP  &  ' + str(round(res_es_00_vgl, 3)) + '&  ' + '& ' + str(
        round(crps_00_res, 3)) + '& (' + str(round(p_vgl_00_es, 3)) + ', - )' + r"\\")
    print('I-Raw  &  ' + str(round(res_es_00_vgl_raw, 3)) + '&  ' + '& ' + str(
        round(crps_00_res_raw, 3)) + '& (' + str(round(p_vgl_00_es_raw, 3)) + ',' + str(
        round(p_vgl_crps, 3)) + ')' + r"\\")

print('Finished calculatios')
