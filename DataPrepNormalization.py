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



################ SECTION 4  DATA PREPRATION AND NORMALIZATION  ############################################
"""   SECTION 4   DATA PREPRATION AND NORMALIZATION
1. Create % Change in Adjusted close by return days based on the parameters passed in n
2. Calculate Moving_Average_Of_ADJ_Close_Return_for_Interval_N based on intervals
3. Remove all non normalized columns and only keep the calculated columns

"""
def addAdjClosePercentChangeAndMovingAvgOfReturn(dataframe, adjclose, returns, n):
    """
    Pick up the Ticker Symbol from AdjClose_Ticker means if you are analyzing for XOM 
    (dataset.insert(0,out where out =XOM in this case) so form AdjClose_XOM 
    Pick XOM then to it add the word "_ADJ_CL_PCT_CHG_DAY" followed by
    days range which is 
    defined in delta = range(2, 5) this is the value received in "n"
    Let Say n = 2,3,4 
    Date	Open	High	Low	Close	Volume	Adj Close
    12/31/2013	100.489998	101.389999	100.43	101.199997	8509600	91.899766
    12/30/2013	101.529999	101.550003	100.309998	100.309998	9007900	91.091558
    12/27/2013	101.239998	101.739998	100.989998	101.510002	10209000	92.181282
    12/26/2013	99.419998	101.029999	99.379997	100.900002	9531200	91.62734
    12/24/2013	98.330002	99.440002	98.330002	99.220001	4168300	90.101731
    12/23/2013	99	99.290001	98.389999	98.510002	10127600	89.456981
    12/20/2013	99.389999	99.599998	98.599998	98.68	23331000	89.611356
    Then
    XOM_DELTA_2 = (91.899766 - 92.181282)/92.181282=-0.003054 -- % Change in 2 days
    XOM_DELTA_3 = (91.899766 - 91.62734)/91.62734=0.002973196 -- % Change in 3 days
    XOM_DELTA_4 = (91.899766 - 90.101731)/90.101731=0.01995561 -- % Change in 4 days
    """
    New_Column_Feature_To_Add = adjclose[9:] + "_ADJ_CL_PCT_CHG_DAY_" + str(n)
    dataframe[New_Column_Feature_To_Add] = dataframe[adjclose].pct_change(n)
    """
    Get the name of the Ticker from Return_Ticker Name example XOM from 'Return_XOM'
    Then calculate the Moving Average of return's (daily % return of Adj Close )
    for intervals received from the value of n (n=2,3,4) and sdd it to a new column in
    Dataframe 
    """
#    print("returns ",returns[7:])
    Moving_Average_Of_ADJ_Close_Return_for_Interval_N = returns[7:] + "_Mov_Agv_Day_Interval_" + str(n)
    dataframe[Moving_Average_Of_ADJ_Close_Return_for_Interval_N] =dataframe[returns].rolling(n).mean()

    
def keepCalculatedColumnsOnly(datasets):
    """
    This method removes  (anything upto AdjClose) original columns
    of "Open,High,Low,Close, AdjClose) 
    
    It will first drop the First data set that is in this case XOM and will store the 
    remaining datasets in a new dataset called dataset_subset_rest
    
    Find the column name anything that has "AdjClose"  by using the followin
    tt = (Individual_dataset.columns[Individual_dataset.columns.str.startswith('AdjClose_')])
    tt =''.join(map(str,tt))
    Once the name is found then find the location of the column using
    pos_adj_cls=Individual_dataset.columns.get_loc(tt)
    After finding the location of the column add 1 as you want to select all the coulmns(calculated columns)
    after  the AdjClose columns 
    Do the same thing for the first data set (XOM) and then join the two .
    Now join/merge the Y and X  this will provide XOM, BRENT etc with XOM in first 
    and with only with the relevant calculated columns
    datasets[0].iloc[:,pos_adj_cls_for_ticker_to_predict:].join(dataset_subset_rest, how = 'outer')
    
    """
    for Individual_dataset in datasets[:1]:
        tt = (Individual_dataset.columns[Individual_dataset.columns.str.startswith('AdjClose_')])
#        print("TT ",tt)
        tt =''.join(map(str,tt))
#        print("TT ",tt)
#        print(" Location of ",tt,Individual_dataset.columns.get_loc(tt))
        pos_adj_cls_for_ticker_to_predict=Individual_dataset.columns.get_loc(tt)
    pos_adj_cls_for_ticker_to_predict = pos_adj_cls_for_ticker_to_predict+1
    
    dataset_subset_rest = []
    for Individual_dataset in datasets[1:]:
        tt = (Individual_dataset.columns[Individual_dataset.columns.str.startswith('AdjClose_')])
        tt =''.join(map(str,tt))
#        print("TT ",tt)
#        print(" Location of ",tt,Individual_dataset.columns.get_loc(tt))
        pos_adj_cls=Individual_dataset.columns.get_loc(tt)
        dataset_subset_rest.append( Individual_dataset.iloc[:,(pos_adj_cls+1):])
        
#    print("Position of 1st Adj Close ",int(pos_adj_cls_for_ticker_to_predict))    
    calculatedDataSetColumns =datasets[0].iloc[:,pos_adj_cls_for_ticker_to_predict:].join(dataset_subset_rest, how = 'outer')

    return  calculatedDataSetColumns
    
    
def Find_NaN(PortfolioDataSets):
    """
    count number of NaN in dataframe
    1. Count the total number of rows between the start date and end date
      if some dates are missing it will count those missing
      dates and add to the number of rows (let say it counts 22 rows
      and 9 missing rows so total Rows =31)
      So Total Number of Rows is dataframe.shape[0] =31
    2. Count the total number of column excluding the index column
     that is the dates , let say the total columns =14
     
    3. Then the data set should have total  31 * 14 = 434 data points
    
    4. Now count the total data points available in the dataframe excluding the
     missing rows and means in this case it will be 22 rows * 14 column = 226 data 
     point which is given by dataframe.count() and then do a sum of each of 
     these counts dataframe.count().sum() = 226
     
     5. So missing data points are 434-226 = 208
    
    """
    return (PortfolioDataSets.shape[0] * PortfolioDataSets.shape[1]) - PortfolioDataSets.count().sum()

    
def Remove_NaN(dataset, MoveUp, delta, back):
    """
    Moving up data by using the back value and removing NaN from Datasets
    """
    
    maxDaysUp = max(MoveUp)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        for days in MoveUp:
            newcolumn = column + str(days)
            dataset[newcolumn] = dataset[column].shift(days)

    return dataset.iloc[maxDaysUp:-1,:]
    




    
    
