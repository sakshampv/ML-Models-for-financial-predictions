import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy.stats import kurtosis
from scipy import stats
from os import listdir
from scipy.stats import skew
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# Importing the dataset


df_train = pd.read_excel (r'./ML_Data/TrainingData2.xlsx', sheetname = "12_MONTH_LAG")

#print(df_train.head(5))
#dataset = df_train.values[:,2:]

y = df_train['PX_LAST']
#X = df_train.drop(columns = ['PX_LAST', 'Stock Name', 'Date'], inplace = False)

# =============================================================================
# WW = X.copy()
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range = (0, 1))
# X = sc.fit_transform(X)
# X = pd.DataFrame(X,columns = WW.columns)
# =============================================================================

X = df_train
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train = X_train.drop(columns = ['PX_LAST', 'Stock Name', 'Date'], inplace = False)
X_test = X_test.drop(columns = ['PX_LAST', 'Stock Name', 'Date'], inplace = False)
stocknames = X_test['Stock Name']
dates  = X_test['Date']
# =============================================================================
# X_train = X
# Y_train = y
# 
# 

X_train, X_val, Y_train, Y_val  = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 0)
# 
# =============================================================================
#df2 = pd.read_excel (r'./ML_Data/ValidationData2.xlsx', sheetname = "12_MONTH_LAG")

# =============================================================================
# Y_test = df2['PX_LAST']
# X_test = df2.drop(columns = ['PX_LAST', 'Stock Name', 'Date'], inplace = False)
# 
# 
# =============================================================================

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1,5e-1,5e-2, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0,1e-2, 1e-3, 1e-1, 1, 2, 5, 7, 10, 50, 100, 150, 200, 500],
             'reg_lambda': [0, 1e-1,1e-2, 5e-2,5e-1, 1, 5, 10, 20, 50, 100],
             'learning_rate' :[1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1, 8e-1, 1,2],
             'min_data_in_leaf' : sp_randint(10, 200),
             'max_depth' : [3,5,10,15,20,25,30,40] }
             


fit_params={"early_stopping_rounds":20, 
            "eval_metric" : 'mae', 
        
            "eval_set" : [(X_val,Y_val)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100000,
            'n_estimators' : 200000,
            'categorical_feature': 'auto'}


from sklearn.metrics import r2_score



n_HP_points_to_test = 400




import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMRegressor(max_depth=-1, random_state=314, silent=True, metric='mae', n_jobs=-1, n_estimators = 200000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=15,
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)
#-------------------------------------------------------------------------
# ATTENTION:: CODE FOR RANDOMISED GRIDSEARCH
# UNCOMMENT BELOW TO RUN RANDOMISED GRIDSEARCH AND THEN COPY THE PARAMETERS PRINTED ON SCREEN INTO OPT_PARAMS
#-----------------------------------------------------------------------------------
# =============================================================================
# gs.fit(X_train, Y_train, **fit_params)
# 
# print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
# 
# 
# =============================================================================
################################################################################################################
clf_final = lgb.LGBMRegressor(**clf.get_params())
#set optimal parameters
 #  {'colsample_bytree': 0.8655532276086073, 'learning_rate': 0.01, 'max_depth': 40, 'min_child_samples': 497, 'min_child_weight': 0.001, 'min_data_in_leaf': 121, 'num_leaves': 45, 'reg_alpha': 5, 'reg_lambda': 0.5, 'subsample': 0.6809523933451751}
opt_params = {'colsample_bytree': 0.8655532276086073, 'learning_rate': 0.01, 'max_depth': 40, 'min_child_samples': 497, 'min_child_weight': 0.001, 'min_data_in_leaf': 121, 'num_leaves': 45, 'reg_alpha': 5, 'reg_lambda': 0.5, 'subsample': 0.6809523933451751} 
#opt_params = {'colsample_bytree': 0.5648879638105521, 'learning_rate': 0.5, 'max_depth': 20, 'min_child_samples': 307, 'min_child_weight': 0.1, 'min_data_in_leaf': 55, 'num_leaves': 10, 'reg_alpha': 100, 'reg_lambda': 0.1, 'subsample': 0.9071147321063562} 
clf_final.set_params(**opt_params)

clf_final.fit(X_train, Y_train, **fit_params)


Y_pred2 = clf_final.predict(X_train)
print(mean_absolute_error(Y_train, Y_pred2))
#Y_test = Y_test.values
cnt = 0
Y_train = Y_train.values
for i in range(len(Y_train)):
    if Y_train[i]*Y_pred2[i] >=0:
        cnt += 1
        
print("Directional Accuracy (Train): "+ str(cnt/len(Y_train)*100))


print("R^2 (Train): "+ str(r2_score(Y_train, Y_pred2)))
        
print("Correlation (Train): "+ str(np.corrcoef(Y_pred2, Y_train)[0,1]))



Y_pred = clf_final.predict(X_test)

Y_pred['Stock Name'] = stocknames
Y_pred['Date'] = dates

Y_pred.to_csv("Predicted_Returns.csv")

# =============================================================================
# sc.fit_transform(y_pred)
# sc.fit_transform(y)
#             
# =============================================================================
from sklearn.metrics import mean_absolute_error

  
print(mean_absolute_error(Y_test, Y_pred))
Y_test = Y_test.values
cnt = 0
for i in range(len(Y_test)):
    if Y_test[i]*Y_pred[i] >=0:
        cnt += 1
        
print("Directional Accuracy (Test): "+ str(cnt/len(Y_test)*100))


print("R^2 (Test): "+ str(r2_score(Y_test, Y_pred)))
        
print("Correlation (Test): "+ str(np.corrcoef(Y_pred, Y_test)[0,1]))




