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

import pandas as pd
import pandas_datareader as pdd
import os
#import datetime
################ SECTION 3  GET DATA SETS  ############################################
"""      SECTION 3  GET DATA SETS
1. Get portfolio index data
2. Get data for stock that will be backtested /predicted

"""
def getPortfolioData(stock_file_directory,Index_Comp_Start_Date,Index_Comp_End_Date,Indexing_Ticker): 
    """
    Collect Index ticker stored in Indexing_Ticker and store it in a dataframe.
    Here I am getting data between specific dates and not complete dataset that has been 
    downloaded from yahoo or quandal
    """
    portfolio=[]
    
#    portfolio =pd.DataFrame()
##    
    dates = pd.date_range(Index_Comp_Start_Date,Index_Comp_End_Date)
    Index_Ticker = pd.DataFrame(index=dates)
#    print("Indexing_Ticker ",Indexing_Ticker)
#   
    
    for ticker in Indexing_Ticker:
#        ticker=ticker.append()
#    print("SHUBRA XX =",ticker)
        fn=os.path.join(stock_file_directory, "{}.csv".format(str(ticker)))
        df = pd.read_csv(fn, index_col='Date',parse_dates=True)
        df1 = Index_Ticker.join(df)
        portfolio.append(df1)
    

    return portfolio




def downloadStockToPredict(symbol, Analysis_start_date, Analysis_end_date,stock_file_directory):
    """
    Get the Stock data that will be analyzed and predicted .
    First look in the directory if the file exists if not then 
    directly from yahoo finance using pandas get_data_yahoo function
    
    """

    try:
        dates = pd.date_range(Analysis_start_date,Analysis_end_date)
        stockToPredict = pd.DataFrame(index=dates)

        fn=os.path.join(stock_file_directory, "{}.csv".format(str(symbol)))

        stockToPredict_temp = pd.read_csv(fn, index_col='Date',parse_dates=True)
#        print("SHUBRA  stockToPredict_temp",stockToPredict_temp)
        stockToPredict = stockToPredict.join(stockToPredict_temp)
#        stockToPredict.columns.values[-1] = 'AdjClose'
#        stockToPredict.columns = stockToPredict.columns + '_'+ symbol
#        stockToPredict['Return_'+symbol] = stockToPredict['AdjClose_'+symbol].pct_change()
#        print("SHUBRA Predicted Stock ",stockToPredict) 
    except:
#        print("No Data Found for ", symbol)
    
        stockToPredict =   pdd.data.get_data_yahoo(symbol, Analysis_start_date, Analysis_end_date)
    
        stockToPredict.columns.values[-1] = 'AdjClose'
        stockToPredict.columns = stockToPredict.columns + '_'+ symbol
        stockToPredict['Return_'+symbol] = stockToPredict['AdjClose_'+symbol].pct_change()
    return stockToPredict

