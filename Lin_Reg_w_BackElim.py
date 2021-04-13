# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 22:26:17 2019

@author: Hp
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df_train = pd.read_excel (r'./ML_Data/TrainingData2.xlsx')
y = df_train['PX_LAST']
X = df_train.drop(columns = ['PX_LAST', 'Stock Name', 'Date'], inplace = False)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)
Y_train = y


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
reg = regressor.fit(X_train, Y_train)



df_test = pd.read_excel (r'./ML_Data/TestData2.xlsx')

X_test = df_test.drop(columns = ['PX_LAST', 'Stock Name', 'Date'], inplace = False)
#X_test = X_train
X_test = sc.fit_transform(X_test)
Y_test = df_test['PX_LAST']
#Y_test = Y_train
#_test = sc.transform(Y_test)

Y_pred = regressor.predict(X_test)
#_pred =sc.transform(Y_pred)
print(np.corrcoef(Y_pred, Y_test))

from sklearn.metrics import r2_score
print(r2_score(Y_test, Y_pred))



from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_test, Y_pred))
cnt = 0
for i in range(len(Y_test)):
    if Y_test[i]*Y_pred[i] >=0:
        cnt += 1
        
print(cnt/len(Y_test)*100)

global deleted


import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    global deleted
    deleted = []
    numVars = len(x[0])
    for i in range(0, numVars):
        print("yes")
        regressor_OLS = sm.OLS(Y_train, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    x = np.delete(x, j, 1)
                    print("Deleted")
                    print(j)
                    deleted += [j]
    regressor_OLS.summary()
    return x
 
    
SL = 0.0005

X_train = np.append(arr = np.ones((len(X_train), 1)).astype(int), values = X_train, axis = 1)

X_Modeled = backwardElimination(X_train, SL)

X_train, X_test, Y_train, Y_test = train_test_split(X_Modeled, Y_train, test_size = 0.2, random_state = 0)

reg = regressor.fit(X_train, Y_train)
#X_test =  np.append(arr = np.ones((len(X_test), 1)).astype(int), values = X_test, axis = 1)
Y_pred = regressor.predict(X_test)
print(np.corrcoef(Y_pred, Y_test))
print(r2_score(Y_test, Y_pred))
print(mean_absolute_error(Y_test, Y_pred))


Y_test = Y_test.values
cnt = 0
for i in range(len(Y_test)):
    if Y_test[i]*Y_pred[i] >=0:
        cnt += 1
        
print(cnt/len(Y_test)*100)




