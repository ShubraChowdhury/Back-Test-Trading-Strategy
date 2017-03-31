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

from sklearn import  ensemble,neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score,r2_score,make_scorer
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.qda import QDA
from sklearn.svm import SVC

################ SECTION 7  CALL THE CLASSIFICATION MODEL #####################    
"""
1. Call the type of classification I would like to use
2. This in turn will call the required Classification method

"""


################ SECTION 9 CLASSIFICATION FUNCTIONS############################  
    


  

    
def performRFClass(X_train, y_train, X_test, y_test):
    """
    Random Forest Binary Classification
    """
    clf = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1)
#    parameters = [{'n_estimators':[10],'criterion':['entropy'],'min_weight_fraction_leaf':[0.5],'n_jobs':[-1]}]

##    # Make an appropriate scoring function
#    #scoring_function = make_scorer(performance_metric(), greater_is_better=False)
#    scoring_function = make_scorer(r2_score, greater_is_better=True)

##    # Make the GridSearchCV object
#    clf = GridSearchCV(clf, parameters,scoring=scoring_function, cv=10)
#    print("Shape X_train:",X_train.shape)
#    print("Shape y_train:",y_train.shape)
#    print("Shape X_test:",X_test.shape)
#    print("Shape y_test:",y_test.shape)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
#    plotLearningPerformance(X_train,y_train,X_test,y_test,10,8,clf,'RandomForest')
    
    return accuracy
        
def performKNNClass(X_train, y_train, X_test, y_test):
    """
    KNN binary Classification
    """
    clf = neighbors.KNeighborsClassifier()
    
    #print("performKNNClass  ",clf.get_params().keys())
    parameters = [{'n_neighbors':[20],'weights':['distance'],'algorithm':['auto'],'n_jobs':[-1]}]

    scoring_function = make_scorer(r2_score, greater_is_better=True)
    clf = GridSearchCV(clf, parameters,scoring=scoring_function, cv=10)

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    auc = roc_auc_score(y_test, clf.predict(X_test))
    print("Area Under Curve =" , auc)
    
    f, ax = plt.subplots(figsize=(8,8))
    ax.plot(y_test,label="test")

    
    return accuracy

def performSVMClass(X_train, y_train, X_test, y_test):
    """
    SVM binary Classification
    """
    clf = SVC()
#    print("performSVMClass  ",clf.get_params().keys())
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy
    
def performAdaBoostClass(X_train, y_train, X_test, y_test, parameters):
    """
    Ada Boosting binary Classification
    """
    n = parameters[0]
    l =  parameters[1]
    clf = AdaBoostClassifier(n_estimators = n, learning_rate = l)
#    print("performAdaBoostClass  ",clf.get_params().keys())
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy
    
def performGTBClass(X_train, y_train, X_test, y_test):
    """
    Gradient Tree Boosting binary Classification
    """
    clf = GradientBoostingClassifier(n_estimators=150)
#    print("performGTBClass  ",clf.get_params().keys())
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy

def performQDAClass(X_train, y_train, X_test, y_test):
    """
    QDA Classification
    """
    clf = QDA()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy

