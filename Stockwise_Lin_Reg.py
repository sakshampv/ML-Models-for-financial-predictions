from matplotlib import pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf
from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

#df_train = pd.read_excel (r'./ML_Data/TrainingData2.xlsx', sheetname = "12_MONTH_LAG")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
qq = pd.read_csv('otp.csv')
qq = qq.drop(columns = ['Unnamed: 0'], inplace = False)
qq['Stock Name'] = df_train['Stock Name']
qq['PX_LAST'] = y.values
# =============================================================================
# 
# y = df_train['PX_LAST']
# X = df_train.drop(columns = ['PX_LAST', 'Stock Name', 'Date'], inplace = False)
# X_train = sc.fit_transform(X)
# X_train = pd.DataFrame(X_train, index = X.index, columns = X.columns)
# 
# X_train['PX_LAST'] = y.values
# X_train['Stock Name'] = df_train['Stock Name']
# =============================================================================

import statsmodels.formula.api as sm


train_corr  = []
test_corr = []
train_r2 = []
test_r2 = []
train_dir = []
test_dir = []
train_mae = []
test_mae = []
num_feat = []
stock_names = []

def backwardElimination(x, sl):
    deleted = []
    numVars = len(x.columns.values)
    #print(numVars)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y_train, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        #print(maxVar)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    if j< len(x.columns):
                      x = x.drop(x.columns[j], axis=1, inplace = False)
                    #print("Deleted")
                    #x = np.delete(x, j, 1)

    #with open('24_parameters_summary.csv', 'w') as fh:                
     #   fh.write((regressor_OLS.summary()).as_csv())
    return x


from sklearn.linear_model import LinearRegression

    
SL = 0.01
from sklearn.model_selection import train_test_split

a = qq.copy()
b = a.groupby('Stock Name').get_group('AMZN UW Equity')

c = np.array(a.groupby('Stock Name').mean().index.values, dtype = 'str')


for stock in c:
    b  = a.groupby('Stock Name').get_group(stock)
    Y_train = b['PX_LAST']
    if( len(Y_train.values)<96):
        continue
    X_train = b.drop(columns = ['PX_LAST', 'Stock Name'], inplace = False)
    #X_train.insert(0,'const', 1)
    X_Modeled = backwardElimination(X_train, SL)
    X_train, X_test, Y_train, Y_test = train_test_split(X_Modeled, Y_train, test_size = 0.3, random_state = 0)
    regressor = LinearRegression()
    if(len(list(X_train.columns)) == 0):
        continue
    reg = regressor.fit(X_train, Y_train)
    print(str(stock) + ' Done')
    Y_pred = regressor.predict(X_train)
    train_corr += [np.corrcoef(Y_train, Y_pred)[0,1]]
    train_r2 += [r2_score(Y_train, Y_pred)]
    train_mae += [mean_absolute_error(Y_train, Y_pred)]
    cnt = 0
    Y_train = Y_train.values
    for i in range(len(Y_train)):
     if Y_train[i]*Y_pred[i] >=0:
        cnt += 1
    train_dir += [(cnt/len(Y_train))*100]
    stock_names += [str(stock)]
    
    num_feat += [len(list(X_Modeled.columns))]
    Y_pred = regressor.predict(X_test)
    test_corr += [np.corrcoef(Y_test, Y_pred)[0,1]]
    test_r2 += [r2_score(Y_test, Y_pred)]
    test_mae += [mean_absolute_error(Y_test, Y_pred)]
    cnt = 0
    Y_test = Y_test.values
    for i in range(len(Y_test)):
     if Y_test[i]*Y_pred[i] >=0:
        cnt += 1
    test_dir += [(cnt/len(Y_test))*100]
    
    
    
    
final_results = pd.DataFrame()

final_results['Stock Name'] = stock_names
final_results['Number of Feartures Selected'] = num_feat
final_results['Train R^2'] = train_r2
final_results['Train Correlation'] = train_corr
final_results['Train MAE'] = train_mae
final_results['Train Directional Accuracy'] = train_dir

final_results['Test R^2'] = test_r2
final_results['Test Correlation'] = test_corr
final_results['Test MAE'] = test_mae
final_results['Test Directional Accuracy'] = test_dir



final_results.to_csv("12moStockwise_Regression_Results_" +str(SL) +  "_.csv")







    
    
    
    
    
    
    
    
    
    
    
    
    
    
    