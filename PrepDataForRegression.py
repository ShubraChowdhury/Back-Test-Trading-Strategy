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

import numpy as np

################ SECTION 6 PREPARE DATA FOR REGRESSION MODELS #################    
""" SECTION 6 PREPARE DATA FOR REGRESSION MODELS 
1. Get the dataset 
2. Use the slpit value and shape of dataset create index
3. Break the dataset in training and testing data
4. Call Multiple regression models and performing plotting

"""

def callRegressionModel(dataset, split):
    """
    Based on the value of split (>0 and < 1) and shape of data set create a index value and 
    based on the index value split the datasets in training and testing 
    
    """
#    print('FROM START ',dataset.columns[:])
    features = dataset.columns[1:]
#    print("features",features)

    index = int(np.floor(dataset.shape[0]*split))
    train, test = dataset[:index], dataset[index:]
 

    output = 'Return_XOM'

    """
    Set the values for X_train , y_train, X_test,y_test
    """
    X_train =train[features]
    y_train =train[output]
    X_test = test[features]
    y_test =test[output]
#    print("y_test ",y_test.head())
#    print("X_test ",X_test.head())
    
    return X_train, y_train, X_test, y_test,output




    
