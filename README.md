# Energy-Schaake
Python functions used in the paper: "From point forecasts to multivariate probabilistic forecasts: The Schaake shuffle for day-ahead electricity price forecasting"
by Oliver Grothe, Fabian Kächele & Fabian Krüger.

## Features
Calculate rank matrix from realized errors
Calculate rank matrix from fitted parametric distribution 
Create density(/ensemlbe)-forecast from a given point forecast, rankmatrix and historic errors with non-parametric margins
Create density(/ensemlbe)-forecast from a given point forecast, rankmatrix and historic errors with parametric margins (t-/normal distribution)


## Technologies
Python version: 3.8
Numpy version: 1.19.2
Scipy verison: 1.6.0

## Usage

```
save Functions_Energy-Schaake.py in project

import Functions_Energy-Schaake as func

## given
errors - realized, standardized errors for dependence learning
sigma  - variance prediction for each hour (e.g. from time series model)
Yp     - 24 dimensional day-ahead point prediction for electricity price 
errordist_normed - realized, standardized errors for marginal distributions


# non-parametric forecast ensemble
rankmatrix=func.get_rankmatrix(errors,param=False)
forecast=func.density_forecast_nonparam(Yp, sigma, _,rankmatrix,errordist_normed)

# parametric forecast ensemble
rankmatrix_parametric=func.get_rankmatrix(errors,param=True)
forecast_parametric=func.density_forecast_param(Yp, sigma, _,rankmatrix_parametric,errordist_normed,dof=0)

```



