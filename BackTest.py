# -*- coding: utf-8 -*-
"""
BackTest.py is the main module which is dependend on 8 other python module.

Depends on the following

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

#import StockTestingWithIndexPrepClassification as tools
import datetime
from GetDataSets import downloadStockToPredict,getPortfolioData
from DataPrepNormalization import addAdjClosePercentChangeAndMovingAvgOfReturn,keepCalculatedColumnsOnly,Find_NaN,Remove_NaN
from PrepDataForClassification import  prepareDataForClassification,check_null_infinite_value
from  CallClassification import performRFClass,performKNNClass,performSVMClass,performGTBClass,performQDAClass
from PrepDataForRegression import callRegressionModel
from RegressionFunctions import exeDecisionTreeRegressor,exeKNeighborsRegressor,exeRandomForestRegressor,exeSVR,exeBaggingRegressor,exeAdaBoostRegressor,exeGradientBoostingRegressor


stock_file_directory='C:/Training/udacity/MachineLearningEngineerNanodegree/P5/Project/Final/BackTestClassRegression/data/'
#stock_file_directory='C:/Training/udacity/MachineLearningEngineerNanodegree/P5/data/add/'


def Back_Test_Ticker():
    """
    Initial parameter settings 
    
    Ticker_To_Analyze This the ticker that we would like to analyze
    Indexing_Ticker This is the Index Tickers that we will build our models to comapre with
    Analysis_start_date This is the start date for the ticker data 
    Analysis_end_date This is the end date for the ticker data)
    Index_Comp_Start_Date This is the start date for the Index ticker data
    Index_Comp_End_Date This is the end date for the Index ticker data
    Train_start_Date Training will be performed on the datasets before this date
    MoveUp--> Shift up by the range mentioned in the range to remove any NaN     
    """
    Ticker_To_Analyze ='XOM'
    Indexing_Ticker = ['WTI','BRENT','VIX']
    Analysis_start_date = datetime.datetime(2010,1,1)
    Analysis_end_date = datetime.datetime(2017,1,10)
    Index_Comp_Start_Date = datetime.datetime(2010,1,1)
    Index_Comp_End_Date = datetime.datetime(2017,1,10)
    Train_start_Date = datetime.datetime(2016, 3, 20)
    MoveUp = range(2, 3)
    


    """
    getPortfolioData is called to download Index data for the specific date ranges , in this case it is XOM data for date range
    between 10th Jan 2017 and 1st Jan 2010
    """
    Ticker_To_Predict = downloadStockToPredict(Ticker_To_Analyze, Analysis_start_date, Analysis_end_date,stock_file_directory)
    
    """
    downloadStockToPredict is called to download Index data for the specific date ranges , in this case it is VIX, BRENT and WTI 
    data for date range
    between 10th Jan 2017 and 1st Jan 2010
    """    
    Index_Compare = getPortfolioData(stock_file_directory,Index_Comp_Start_Date,Index_Comp_End_Date,Indexing_Ticker)

    """
    Adding XOM as the first data set in the frame
    """
    Index_Compare.insert(0, Ticker_To_Predict) 
    
    """
    After inserting XOM now I am renaming the dataset to Portfolio_Analysis_Dataset
    """
    Portfolio_Analysis_Dataset=Index_Compare
    
    """
    Some of my validation and crosscheck not required for Final run and I will comment this
    """
#    print("Printing Data for ",Ticker_To_Analyze)
#    print(Ticker_To_Predict.tail())
    
    """
    Add additional 4 features  features for <Ticker>_ADJ_CL_PCT_CHG_DAY_<range> 
    and <Ticker>_Mov_Agv_Day_Interval_<Range>
       
    """
    
    Add_Features = range(2, 6)
#    print ('Max Delta days accounted: ', max(Add_Features))
    Portfolio_dataset,count_NaN=check_AddFeatures(Portfolio_Analysis_Dataset,Add_Features,Ticker_To_Analyze,MoveUp)


    """
    Calling prepareDataForClassification to prepare training and testing data sets to be used By CLASSIFICATION MODELs
    
    """
    X_train, y_train, X_test, y_test  = prepareDataForClassification(Portfolio_dataset, Train_start_Date,Ticker_To_Analyze)
    
    """
    Checking for NaN and Infinity in daytasets To make sure if dataset is clean or not for analysis
    
    """
    print('')
#    print 'Maximum days Move Up :', max(MoveUp)
    print("NaN Infinity checker :", check_null_infinite_value(X_train, y_train, X_test, y_test)  )         
    print('')
    
    """
    Calling Multiple Model Of Classification
    """  
    print('========== Classification Results ===============================')
    print("Classification Randon Forest Score :", performRFClass(X_train, y_train, X_test, y_test))
    print("Classification KNN Score :", performKNNClass(X_train, y_train, X_test, y_test))
    print("Classification SVM Score :", performSVMClass(X_train, y_train, X_test, y_test))
    print("Classification GradientBoostingClassifier Score :", performGTBClass(X_train, y_train, X_test, y_test))
    print("Classification QDA Score :", performQDAClass(X_train, y_train, X_test, y_test))
    print('')    
            
        
    """
    Calling callRegressionModel to prepare training and testing data sets to be used By REGRESSION MODELS
    
    """    
    X_train_reg, y_train_reg, X_test_reg, y_test_reg,output = callRegressionModel(Portfolio_dataset,.6)
    print('')
    print("Shape of X_train_reg :",X_train_reg.shape )
    print("Shape of y_train_reg :",y_train_reg.shape )
    print("Shape of X_test_reg :",X_test_reg.shape )
    print("Shape of y_test_reg :",y_test_reg.shape )
    
    """
    Calling Multiple Model For Regression
    """    
    print('========== Regression Results ===============================')
    print ('GridSearchCV Tuning on Decission Tree Regressor ', exeDecisionTreeRegressor(X_train_reg, y_train_reg, X_test_reg, y_test_reg))
    print('') 
    print ('KNN Regression Score  ', exeKNeighborsRegressor(X_train_reg, y_train_reg, X_test_reg, y_test_reg))
    print('') 
    print ('RandomForest Regression Score  ',exeRandomForestRegressor(X_train_reg, y_train_reg, X_test_reg, y_test_reg))
    print('') 
    print ('SVR Regression Score  ',exeSVR(X_train_reg, y_train_reg, X_test_reg, y_test_reg))
    print('') 
    print ('Bagging Regression Score  ',exeBaggingRegressor(X_train_reg, y_train_reg, X_test_reg, y_test_reg))
    print('') 
    print ('AdaBoost Regression Score  ',exeAdaBoostRegressor(X_train_reg, y_train_reg, X_test_reg, y_test_reg))
    print('') 
    print ('GradientBoosting Regression Score  ',exeGradientBoostingRegressor(X_train_reg, y_train_reg, X_test_reg, y_test_reg))
 
   
def check_AddFeatures(Portfolio_Analysis_Dataset,Add_Features,Ticker_To_Analyze,MoveUp):
    for Individual_dataset in Portfolio_Analysis_Dataset:
        columns = Individual_dataset.columns    
        adjclose = columns[-2]
        returns = columns[-1]
        for n in Add_Features:
            addAdjClosePercentChangeAndMovingAvgOfReturn(Individual_dataset, adjclose, returns, n)
        
    Portfolio_dataset = keepCalculatedColumnsOnly(Portfolio_Analysis_Dataset)

    print ('Initial Size of portfolio: ', Portfolio_dataset.shape)
    print ('Total Percentage of Non Numeric and Infinity in Initial Portfolio: ', (Find_NaN(Portfolio_dataset)/float(Portfolio_dataset.shape[0]*Portfolio_dataset.shape[1]))*100, '%')
    
    """
    REMOVING NaN, Missing Data and Removing Infinite 
    
    1. Using interpolate fill the missing data for dates in between dates
    2. Use Fillna to fill in the missing data after interpolation
    """
    Portfolio_dataset = Portfolio_dataset.interpolate(method = 'time')
    print("Total Number Of Datapoint Interpolated ",Portfolio_dataset.count().sum())
    print ('Post interpolation Percentage of Non Numeric and Infinity: ', (Find_NaN(Portfolio_dataset)/float(Portfolio_dataset.shape[0]*Portfolio_dataset.shape[1]))*100, '%')
    Portfolio_dataset = Portfolio_dataset.fillna(Portfolio_dataset.mean())
    print ('Post interpolation & Fillna Percentage of Non Numeric and Infinity: ', (Find_NaN(Portfolio_dataset)/float(Portfolio_dataset.shape[0]*Portfolio_dataset.shape[1]))*100, '%')
#    print("interpolate fill NA ",Portfolio_dataset.count().sum())
    back = -1
    """
    Pass the Ticker Name , in this case it is Return_XOM
    """
    Portfolio_dataset = Portfolio_dataset.rename(columns={'Return_'+Ticker_To_Analyze:'Return_Out'})
    Portfolio_dataset.Return_Out = Portfolio_dataset.Return_Out.shift(back)
    Portfolio_dataset = Portfolio_dataset.rename(columns={'Return_Out':'Return_'+Ticker_To_Analyze})
    Portfolio_dataset = Remove_NaN(Portfolio_dataset, MoveUp, Add_Features, back)
#    print 'Number of NaN after temporal shifting: ', Find_NaN(Portfolio_dataset)
    count_NaN =Find_NaN(Portfolio_dataset)
    print ('Final  Size of portfolio: ', Portfolio_dataset.shape)
    return Portfolio_dataset,count_NaN
    
    
if __name__ == "__main__":
    Back_Test_Ticker()     