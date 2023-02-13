# The Schaake shuffle for day-ahead electricity price forecasting
This repository contains the implementation of functions used in the [paper](https://arxiv.org/abs/2204.10154): 

***From point forecasts to multivariate probabilistic forecasts: The Schaake shuffle for day-ahead electricity price forecasting***

by Oliver Grothe, Fabian Kächele & Fabian Krüger.

##### Abstract  
*Modeling price risks is crucial for economic decision making in energy markets. Besides the risk associated with a single price, the dependence structure of multiple prices is often relevant. We therefore propose a generic and easy-to-implement method for generating multivariate probabilistic forecasts based on univariate point forecasts of day-ahead electricity prices. While each univariate point forecast refers to one of the day's 24 hours, the multivariate forecast distribution models dependencies across hours. The proposed method is based on simple copula techniques and an optional time series component. We illustrate the method for five benchmark data sets. 
Furthermore, we demonstrate an example for constructing realistic prediction intervals for the weighted sum of consecutive electricity prices as needed for pricing individual load profiles.* 

## Features
- Create multivaraite density (/ensemble)-forecast from a series of given point forecasts and realizations
  - with non-parametric margins
  - with parametric margins (t-/normal distribution)
  
- Calculate rank matrix for dependence modeling
  -  from realized, past errors  
  -  from fitted parametric distribution  
  
- Reproduce results from the paper


## Usage
#### Requirements
- Python 3.7 or higher
- R version 4.0.1  and rugarch 1.4.4
- Packages specified in environment.yml

#### STAND ALONE USAGE
##### Given
- ***errors***: Series of realized errors from the point forecasting model
- ***y_pred***: 24 dimensional day-ahead point prediction for electricity price

```python
from EnergySchaake import *

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
#### PAPER EXPERIMENTS

Set up virtual environment:
```
conda env create --name envname --file=environment.yml
```
Install epftoolbox:
```
git clone https://github.com/jeslago/epftoolbox.git
cd epftoolbox
pip install .
```

Save all skripts in the corresponding folder of the environment on your system.

Run script for each market:
```
python experiments_paper.py --dataset 'DE'
```

Run Load-Profile Experiment
```
python experiments_paper_load_intervalls.py 
```

Run Load Forecasting Experiment
```
python experiments_paper_load_forecasting.py 
```





