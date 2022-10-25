# The Schaake shuffle for day-ahead electricity price forecasting
This repository contains the implementation of functions used in the [paper](https://arxiv.org/abs/2204.10154): 

***From point forecasts to multivariate probabilistic forecasts: The Schaake shuffle for day-ahead electricity price forecasting***


by Oliver Grothe, Fabian Kächele & Fabian Krüger.

## Technologies
- Python 3.8  
- R 4.0.3
- Numpy 1.19.2  
- Scipy 1.6.0  
- rpy2 2.9.4
- rugarch (R package) 1.4-8


## Features
- Create multivaraite density (/ensemble)-forecast from a series of given point forecasts and realizations
  - with non-parametric margins
  - with parametric margins (t-/normal distribution)
  
- Calculate rank matrix for dependence modeling
  -  from realized, past errors  
  -  from fitted parametric distribution  


## Usage
#### Given data
- ***errors***: Series of realized errors from the point forecasting model
- ***y_pred***: 24 dimensional day-ahead point prediction for electricity price
```python
from EnergyShaake import *

# set parameters for forecast
lenght_error_learning = 90
length_dependence_learning = 90
timeseries_treatment = True
param_margin = False
param_dependence = False

# get forecast of for next day
forecast_density = learn_multivariate_density(y_pred, errors, timeseries_treatment, 
                      lenght_error_learning, length_dependence_learning, 
                        param_margin, param_dependence)
                        
# get only rank matrix of raw errors
data_dependence = errors[-length_dependence_learning:, :].copy()
rankmatrix = get_rankmatrix(errors,param_dependence):

```



