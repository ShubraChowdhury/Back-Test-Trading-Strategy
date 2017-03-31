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
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from sklearn import  ensemble
from sklearn.metrics import mean_squared_error, r2_score,make_scorer
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

#import datetime
from PlotModelCcomplexity import plotLearningPerformance, plotModelComplexity
from  LearnndPerfCalc import learning_curves,model_complexity
################ SECTION 8 REGRESSION FUNCTIONS ##############################    
"""
1.DecisionTreeRegressor
2. KNeighborsRegressor
3. RandomForestRegressor
4.SVR
5.BaggingRegressor
6.AdaBoostRegressor
7.GradientBoostingRegressor
"""

def exeDecisionTreeRegressor(X_train, y_train, X_test, y_test):
    """ Tunes a decision tree regressor model using GridSearchCV on the input data X 
        and target labels y and returns this optimal model. """

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor( max_depth=2,min_samples_leaf=1, min_samples_split=2,splitter='best')

    # Set up the parameters we wish to tune
    parameters = [{'max_depth':(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),'presort':['True']}]

    # Make an appropriate scoring function
    scoring_function = None
    #scoring_function = make_scorer(performance_metric(), greater_is_better=False)
    scoring_function = make_scorer(r2_score, greater_is_better=True)

    # Make the GridSearchCV object
    reg = None
    reg = GridSearchCV(regressor, parameters,scoring=scoring_function, cv=10)

    # Fit the learner to the data to obtain the optimal model with tuned parameters
    reg.fit(X_train, y_train)
    Predicted = reg.predict(X_test)
    print("DecisionTreeRegressor = ",reg.score(X_test, y_test) )
    
    #print "Best model parameter:  " + str(reg.best_params_)
    #print "Best model estimator:  " + str(reg.best_estimator_)
    # Return the optimal model
    print("y_test.index[:] ",y_test.index[:])
    print("Predicted ",Predicted)
    print("y_test ",y_test)
    f, ax = plt.subplots(figsize=(8,8))
    ax.plot(y_test,label=y_test.name[7:])
    ax.plot(y_test.index[:],Predicted,color='red',label="Comparative Port Predicted Value For Test Period")
    ax.plot_date(y_test.index[:],Predicted,color='black')
    ax.set_title('DecisionTreeRegressor')   
    ax.set_xlabel("Date",rotation=0)
    plt.xticks(rotation = 90)
    ax.set_ylabel("Return")
    ax.legend(loc="upper right")
  
    plt.show()
    
    learning_curves(X_train, y_train, X_test, y_test)
    model_complexity(X_train, y_train, X_test, y_test)
    return "DecisionTreeRegressor Best Estimator ", reg.best_estimator_, "DecisionTreeRegressor MSE =", mean_squared_error(y_test,Predicted),"DecisionTreeRegressor R2 =", r2_score(y_test, Predicted)

def exeKNeighborsRegressor(X_train, y_train, X_test, y_test):
    """
    KNN Regression
    """
    clf = KNeighborsRegressor()
    clf.fit(X_train, y_train)
    Predicted = clf.predict(X_test)
    print("KNeighborsRegressor Score = ",clf.score(X_test, y_test) )
    f, ax = plt.subplots(figsize=(8,8))
    ax.plot(y_test,label=y_test.name[7:])
    ax.plot(y_test.index[:],Predicted,color='red',label="Comparative Port Predicted Value For Test Period")
    ax.plot_date(y_test.index[:],Predicted,color='black')
    ax.set_title('K Neighbors Regressor')   
    ax.set_xlabel("Date",rotation=0)
    plt.xticks(rotation = 90)
    ax.set_ylabel("Return")
    ax.legend(loc="upper right")
  
    plt.show()        
#    print("Mean Square ", mean_squared_error(test[output],Predicted))
#    print("R Square ",r2_score(test[output], Predicted))
#    print("output ",output[7:])
#    plotLearningPerformance(X_train,y_train,X_test,y_test,10,8,clf,'KNeighborsRegressor')
#    plotModelComplexity(X_train, y_train, X_test, y_test,clf,'KNeighborsRegressor')
    
    return "KNeighborsRegressor MSE =" ,mean_squared_error(y_test,Predicted),"KNeighborsRegressor R2 =" , r2_score(y_test, Predicted)

def exeRandomForestRegressor(X_train, y_train, X_test, y_test):
    """
    Random Forest Regression
    """
    clf = ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1)
    
    parameters = [{'n_estimators':[20],'criterion':['mse'],'min_weight_fraction_leaf':[0.25],'n_jobs':[-1]}]
    scoring_function = make_scorer(r2_score, greater_is_better=True)
    # Make the GridSearchCV object
    clf = GridSearchCV(clf, parameters,scoring=scoring_function, cv=10)
    
    clf = clf.fit(X_train, y_train)
    Predicted = clf.predict(X_test)
    print("RandomForestRegressor Score = ",clf.score(X_test, y_test) )
    
#    print("y_test ",y_test.name[7:] )
    f, ax = plt.subplots(figsize=(8,8))
    ax.plot(y_test,label=y_test.name[7:])
    ax.plot(y_test.index[:],Predicted,color='red',label="Predicted Port Value ")
    ax.plot_date(y_test.index[:],Predicted,color='black')
    ax.set_title('RF Regressor')   
    ax.set_xlabel("Date",rotation=0)
    plt.xticks(rotation = 90)
    ax.set_ylabel("Return")
    ax.legend(loc="upper right")
    plt.show() 
    
#    plotLearningPerformance(X_train,y_train,X_test,y_test,10,8,clf,'RandomForest')
#    plotModelComplexity(X_train, y_train, X_test, y_test,clf,'RandomForest')
    return y_test.name[7:],"y_test, Random Forest Predicted Mean Square Error =",mean_squared_error(y_test, Predicted), "and Random Forest R-Square =",r2_score(y_test, Predicted)

def exeSVR(X_train, y_train, X_test, y_test):
    """
    SVM Regression
    """

    clf = SVR()
    
    parameters =[{'C': [1, 10, 100, 1000], 'gamma': [1e-1, 1, 1e1], 'kernel': ['rbf','linear', 'poly','sigmoid'],'degree': [3],'epsilon':[0.9]}]
    scoring_function = make_scorer(r2_score, greater_is_better=True)
    # Make the GridSearchCV object
    clf = GridSearchCV(clf, parameters,scoring=scoring_function,cv=10)
    
        
    clf.fit(X_train, y_train)
    Predicted = clf.predict(X_test)
    print("SVR Score = ",clf.score(X_test, y_test) )
    
    f, ax = plt.subplots(figsize=(8,8))
    ax.plot(y_test,label=y_test.name[7:])
    ax.plot(y_test.index[:],Predicted,color='red',label="Predicted Port Value ")
    ax.plot_date(y_test.index[:],Predicted,color='black')
    ax.set_title('SVM Regressor')   
    ax.set_xlabel("Date",rotation=0)
    plt.xticks(rotation = 90)
    ax.set_ylabel("Return")
    ax.legend(loc="upper right")
    
    plt.show()        
    
#    plotLearningPerformance(X_train,y_train,X_test,y_test,10,8,clf,'SV')
#    plotModelComplexity(X_train, y_train, X_test, y_test,clf,'SV')
    
    return "SVR Mean Square Error =",mean_squared_error(y_test,Predicted),"SVR R2 =", r2_score(y_test, Predicted)
    
def exeBaggingRegressor(X_train, y_train, X_test, y_test):
    """
    Bagging Regression
    """
  
    clf = ensemble.BaggingRegressor()
    
    
    clf.fit(X_train, y_train)
    Predicted = clf.predict(X_test)
    print("BaggingRegressor Score = ",clf.score(X_test, y_test) )
    
    f, ax = plt.subplots(figsize=(8,8))
    ax.plot(y_test,label=y_test.name[7:])
    ax.plot(y_test.index[:],Predicted,color='red',label="Comparative Port Predicted Value For Test Period")
    ax.plot_date(y_test.index[:],Predicted,color='black')
    ax.set_title('Bagging Regressor')   
    ax.set_xlabel("Date",rotation=0)
    plt.xticks(rotation = 90)
    ax.set_ylabel("Return")
    ax.legend(loc="upper right")
    
    plt.show()        
    
#    plotLearningPerformance(X_train,y_train,X_test,y_test,10,8,clf,'Bagging')
#    plotModelComplexity(X_train, y_train, X_test, y_test,clf,'Bagging')
    
    return "BaggingRegressor MSE =" ,  mean_squared_error(y_test,Predicted),"BaggingRegressor R2 =", r2_score(y_test, Predicted)  

def exeAdaBoostRegressor(X_train, y_train, X_test, y_test):
    """
    Ada Boost Regression
    """

    clf = ensemble.AdaBoostRegressor()
    clf.fit(X_train,y_train)
    Predicted = clf.predict(X_test)
    print("AdaBoostRegressor Score = ",clf.score(X_test, y_test) )
    
    f, ax = plt.subplots(figsize=(8,8))
    ax.plot(y_test,label=y_test.name[7:])
    ax.plot(y_test.index[:],Predicted,color='red',label="Comparative Port Predicted Value For Test Period")
    ax.plot_date(y_test.index[:],Predicted,color='black')
    ax.set_title('Adaboost Regressor')   
    ax.set_xlabel("Date",rotation=0)
    plt.xticks(rotation = 90)
    ax.set_ylabel("Return")
    ax.legend(loc="upper right")
    
    plt.show()        
    
#    plotLearningPerformance(X_train,y_train,X_test,y_test,10,8,clf,'Ada Boost')
#    plotModelComplexity(X_train, y_train, X_test, y_test,clf,'Ada Boost')
    
    return "AdaBoost MSE =" ,  mean_squared_error(y_test,Predicted),"AdaBoost R2 =", r2_score(y_test, Predicted)

def exeGradientBoostingRegressor(X_train, y_train, X_test, y_test):
    """
    Gradient Boosting Regression
    """
    
    clf = ensemble.GradientBoostingRegressor()
    clf.fit(X_train, y_train)
    Predicted = clf.predict(X_test)
    print("GradientBoostingRegressor Score = ",clf.score(X_test, y_test) )
    
    f, ax = plt.subplots(figsize=(8,8))
    ax.plot(y_test,label=y_test.name[7:])
    ax.plot(y_test.index[:],Predicted,color='red',label="Comparative Port Predicted Value For Test Period")
    ax.plot_date(y_test.index[:],Predicted,color='black')
    ax.set_title('Gradient Boosting Regressor')   
    ax.set_xlabel("Date",rotation=0)
    plt.xticks(rotation = 90)
    ax.set_ylabel("Return")
    ax.legend(loc="upper right")
    
    plt.show()    
    
#    plotLearningPerformance(X_train,y_train,X_test,y_test,10,8,clf,'GradientBoosting')
#    plotModelComplexity(X_train, y_train, X_test, y_test,clf,'GradientBoosting')
    
    return "GradientBoosting MSE =" , mean_squared_error(y_test,Predicted),"GradientBoosting R2 =", r2_score(y_test, Predicted)
    
