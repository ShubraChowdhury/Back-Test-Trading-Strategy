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

from sklearn import preprocessing 
import numpy as np


################ SECTION 5 PREPARE DATA FOR CLASSIFICATION MODELS #############  
"""  SECTION 5 PREPARE DATA FOR CLASSIFICATION MODELS 
1. Prepare training and testing sets for classification
2. Test classification with training data
"""    
    
def prepareDataForClassification(dataset, start_test,Ticker_To_Analyze):
    """
    1. Add a column to the dataset called "Classification"
    2. Assign Return Value of Ticker value to the column "Classification"
    3. If the Classification is >=0 then return 'Days_Return_Gain'
    4. If the Classification is <0  then return 'Days_Return_Loss'
    5. Use sklearn preprocessing.LabelEncoder() ' fit and transform to transform 
    Days_Return_Gain  and Days_Return_Loss to 0 or 1 
    6. All columns except the last column "Classification" is feature and the last 
    column is for testing
    7. All training data belongs to the timeperiod less than the start_test
     8. All testing data belongs to the timeperiod = > than the start_test
    """
    le = preprocessing.LabelEncoder()
    
    dataset['Classification'] = dataset['Return_'+Ticker_To_Analyze]
    
#    dataset.Classification[dataset.Classification >= 0] = 'Days_Return_Gain'
    row_index = dataset.Classification >= 0
    dataset.loc[row_index, 'Classification'] = 'Days_Return_Gain'
    
#    print("row_index   ",row_index, dataset.Classification)
#    dataset.Classification[dataset.Classification < 0] = 'Days_Return_Loss'
    
    row_index1 = dataset.Classification < 0
    dataset.loc[row_index1, 'Classification'] = 'Days_Return_Loss'
    
#    print("row_index   ",row_index1 , dataset.Classification)
    
    dataset.Classification = le.fit(dataset.Classification).transform(dataset.Classification)

    
    features = dataset.columns[1:-1]
#    print("features ",features)
    X_features = dataset[features]    
    y_Classification = dataset.Classification    
    
#    print(" X_features = ", X_features.head())
#    print(" y_Classification = ", y_Classification.head())
    
    X_train = X_features[X_features.index < start_test]
    y_train = y_Classification[y_Classification.index < start_test]    
    
    X_test = X_features[X_features.index >= start_test]    
    y_test = y_Classification[y_Classification.index >= start_test]
    
    return X_train, y_train, X_test, y_test    
    
def check_null_infinite_value(X_train, y_train, X_test, y_test):
       
        
    if np.any(np.isnan(X_train)):
        #print (np.where(np.any(np.isnan(X_train))),"X_train  null found Change the value of lags and Add_Features")
        print ("X_train  null found Change the value of lags and Add_Features")
        
    elif np.any(np.isnan(y_train)):
       # print (y_train,"y_train  null found  Change the value of lags and Add_Features")
        print ("y_train  null found  Change the value of lags and Add_Features")
    elif np.any(np.isnan(X_test)):
      #  print (X_test,"X_test  null found  Change the value of lags and Add_Features")
        print ("X_test  null found  Change the value of lags and Add_Features")
    elif np.any(np.isnan(y_test)):
       # print( y_test,"y_test  null found  Change the value of lags and Add_Features")
        print( "y_test  null found  Change the value of lags and Add_Features")
    else :
        print "Data Set has No  Nan Value " 
        

    

