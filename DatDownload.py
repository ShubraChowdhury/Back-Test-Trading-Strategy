# -*- coding: utf-8 -*-
"""
0. Run the file DataDownload.py to Download Specific Ticker Related Data
1. SECTION 1 -- LEARNING AND PERFORMANCE CALCULATION (LearnndPerfCalc.py)
2. SECTION 2 -- PLOT LEARNING AND MODEL COMPLEXITY (PlotModelCcomplexity.py) --- This is now obselete use LearnndPerfCalc.py instead
3. SECTION 3 -- GET DATA SETS (GetDataSets.py)
4. SECTION 4-- DATA PREPRATION AND NORMALIZATION (DataPrepNormalization.py)
5. SECTION 5 PREPARE DATA FOR CLASSIFICATION MODELS (PrepDataForClassification.py)   
6. SECTION 6 PREPARE DATA FOR REGRESSION MODELS  and Call regression models(PrepDataForRegression.py)
7. SECTION 7  CALL THE CLASSIFICATION MODEL (CallClassification.py)
8. SECTION 8 REGRESSION FUNCTIONS (RegressionFunctions.py)
9. SECTION 9 CLASSIFICATION FUNCTIONS (CallClassification.py)

"""

#import pandas as pd
import pandas_datareader as pd
import quandl
import datetime


start = datetime.datetime(1990, 1, 1)
end = datetime.datetime(2017, 01, 31)

path = 'C://Training/udacity/MachineLearningEngineerNanodegree/P5/Project/Final/BackTestClassRegression/data/'

## XOM 
xom =  pd.data.get_data_yahoo('XOM', start, end)

xom.columns.values[-1] = 'AdjClose'
xom.columns = xom.columns + '_XOM'
xom['Return_XOM'] = xom['AdjClose_XOM'].pct_change()

xom.to_csv(path+'XOM.csv')

## ^VIX 
vix =  pd.data.get_data_yahoo('^VIX', start, end)

vix.columns.values[-1] = 'AdjClose'
vix.columns = vix.columns + '_VIX'
vix['Return_VIX'] = vix['AdjClose_VIX'].pct_change()

vix.to_csv(path+'VIX.csv')


## SPY
sp =  pd.data.get_data_yahoo('SPY', start, end)

sp.columns.values[-1] = 'AdjClose'
sp.columns = sp.columns + '_SPY'
sp['Return_SPY'] = sp['AdjClose_SPY'].pct_change()

sp.to_csv(path+'SPY.csv')

###  WTI  
##https://www.quandl.com/api/v3/datasets/EIA/PET_RWTC_D.csv?api_key=zRHWHkTCcRK2-VzfgKwh


wti = quandl.get("EIA/PET_RWTC_D", trim_start='1990-01-01', trim_end='2017-01-31')


wti.columns.values[-1] = 'AdjClose'
wti.columns = wti.columns + '_WTI'
wti['Return_WTI'] = wti['AdjClose_WTI'].pct_change()

wti.to_csv(path+'WTI.csv')

########## BRENT
##https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D.csv?api_key=zRHWHkTCcRK2-VzfgKwh

brent =  quandl.get("EIA/PET_RBRTE_D", trim_start='1990-01-01', trim_end='2017-01-31')

#oil.columns = oil.columns + '_BRENT'
#oil['Delta_Oil'] = oil['Value_BRENT'].pct_change()

brent.columns.values[-1] = 'AdjClose'
brent.columns = brent.columns + '_BRENT'
brent['Return_BRENT'] = brent['AdjClose_BRENT'].pct_change()
brent.to_csv(path+'BRENT.csv')


### GOOG 
#out =  pd.io.data.get_data_yahoo('GOOG', start, end)
#
#out.columns.values[-1] = 'AdjClose'
#out.columns = out.columns + '_GOOG'
#out['Return_GOOG'] = out['AdjClose_GOOG'].pct_change()
#
#out.to_csv(path+'GOOG.csv')
