

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#df_train = pd.read_excel (r'./ML_Data/TrainingData2.xlsx')
y = df_train['PX_LAST']
X = df_train.drop(columns = ['PX_LAST', 'Stock Name', 'Date'], inplace = False)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X)
X_train = pd.DataFrame(X_train, index = X.index, columns = X.columns)
Y_train = y



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
reg = regressor.fit(X_train, Y_train)




#Y_test = Y_train
#_test = sc.transform(Y_test)

Y_pred = regressor.predict(X_train)
#_pred =sc.transform(Y_pred)
print(np.corrcoef(Y_pred, Y_train))

from sklearn.metrics import r2_score
print(r2_score(Y_train, Y_pred))



from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_train, Y_pred))
cnt = 0
for i in range(len(Y_train)):
    if Y_train[i]*Y_pred[i] >=0:
        cnt += 1
        
print(cnt/len(Y_train)*100)

global deleted


import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    global deleted
    deleted = []
    numVars = len(x.columns.values)
    for i in range(0, numVars):
        print("yes")
        regressor_OLS = sm.OLS(Y_train, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    x = x.drop(x.columns[j], axis=1, inplace = False)
                    #x = np.delete(x, j, 1)
                    print("Deleted")
                    print(j)
                    deleted += [j]
    with open('24_parameters_summary.csv', 'w') as fh:                
        fh.write((regressor_OLS.summary()).as_csv())
    return x
 
    
SL = 0.01

# =============================================================================
# arr = np.ones((len(X_train), 1))
# columns = X_train.columns
# X_train = X_train[['const', columns]]
# X_train['const'] = arr
# =============================================================================
X_train.insert(0,'const', 1)

#X_train = np.append(arr = np.ones((len(X_train), 1)).astype(int), values = X_train, axis = 1)

X_Modeled = backwardElimination(X_train, SL)
#X_Modeled.to_csv("12monthmodifieddata.csv")
X_train, X_test, Y_train, Y_test = train_test_split(X_Modeled, Y_train, test_size = 0.2, random_state = 0)

reg = regressor.fit(X_train, Y_train)
print(regressor)
print(reg)
from statsmodels.api import OLS
# =============================================================================
# with open('24_parameters_summary.txt', 'w') as fh:  
#               fh.write(str(OLS(Y_train, X_Modeled).fit().summary()))
#               
#   
# =============================================================================


Y_pred = regressor.predict(X_train)
print(np.corrcoef(Y_pred, Y_train))
print(r2_score(Y_train, Y_pred))
print(mean_absolute_error(Y_train, Y_pred))

Y_train = Y_train.values
#Y_pred = Y_pred.values

cnt = 0
for i in range(len(Y_train)):
    if Y_train[i]*Y_pred[i] >=0:
        cnt += 1
        
print(cnt/len(Y_train)*100)

            

#X_test =  np.append(arr = np.ones((len(X_test), 1)).astype(int), values = X_test, axis = 1)
Y_pred = regressor.predict(X_test)
print(np.corrcoef(Y_pred, Y_test))
print(r2_score(Y_test, Y_pred))
print(mean_absolute_error(Y_test, Y_pred))

a = []
b = []

#Y_test = Y_test.values
for i in range(len(Y_test)):
    if( Y_test[i] >=0):
        a += [1]
    else:
        a += [0]
        

for i in range(len(Y_pred)):
    if( Y_pred[i] >=0):
        b += [1]
    else:
        b += [0]   
        
print(r2_score(a,b))    

    
print(np.corrcoef(a,b))
print(r2_score(a,b))
print(mean_absolute_error(a,b))
    

Y_test = Y_test.values
cnt = 0
for i in range(len(Y_test)):
    if Y_test[i]*Y_pred[i] >=0:
        cnt += 1
        
print(cnt/len(Y_test)*100)




