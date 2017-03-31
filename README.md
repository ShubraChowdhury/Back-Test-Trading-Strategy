# Back-Test-Trading-Strategy
Back testing of Trading strategy using Machine Learning

Setup
Software and Library:
•	Python 2.7.12 |Anaconda 4.0.0 (64-bit)| (default, Jun 29 2016, 11:07:13) [MSC v.1500 64 bit (AMD64)] on win32
•	scikit-learn version  0.17.1
•	Pandas version 0.18.0
•	NumPy version 1.10.4
•	 SciPy version 0.17.0
•	pandas_datareader version 0.3.0.post
•	quandl api_version = '2015-04-09'


Data
Ticker and data download location:
•	Yahoo (XOM, ^VIX, SPY)
•	Quandl (BRENT and WTI)

Directory Structure :
I have created a directory to download and store data. In my case I have created a directory 
'C://Training/udacity/MachineLearningEngineerNanodegree/P5/Project/Final/BackTestClassRegression/data/'  to store and analyze data. You may have to change or create a directory to store and analyze data.

Main :


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
