



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

df_train2 = pd.read_csv(r'./transformed_training_data.csv') 
#df_train = pd.read_excel (r'./ML_Data/TrainingData2.xlsx')
y = df_train2['PX_LAST']
#X = df_train.drop(columns = ['PX_LAST', 'Stock Name', 'Date'], inplace = False)
X = df_train2.drop(columns = ['PX_LAST'], inplace = False)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.reshape(y.values, (len(y.values),1))
y = sc_y.fit_transform(y)
y = np.reshape(y, (len(y),))
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import AdaBoostRegressor
regressor = AdaBoostRegressor(random_state=0, n_estimators=4000, learning_rate = 0.8)
regressor.fit(X_train, Y_train) 



Y_2 = regressor.predict(X_train)
Y_2 = np.reshape(Y_2, (len(Y_2), 1))
Y_2 = sc_y.inverse_transform(Y_2)
Y_2 = np.reshape(Y_2, (len(Y_2), ))

Y_train = np.reshape(Y_train, (len(Y_train), 1))
Y_train = sc_y.inverse_transform(Y_train)
Y_train = np.reshape(Y_train, (len(Y_train), ))

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
print(np.corrcoef(Y_2, Y_train))
print(r2_score(Y_train, Y_2))
print(mean_absolute_error(Y_train, Y_2))


#Y_train = Y_train.values

cnt = 0
for i in range(len(Y_train)):
    if Y_train[i]*Y_2[i] >=0:
        cnt += 1
        
print(cnt/len(Y_train)*100)






# Predicting a new result
Y_pred = regressor.predict(X_test)
Y_pred = np.reshape(Y_pred, (len(Y_pred), 1))
Y_pred = sc_y.inverse_transform(Y_pred)
Y_pred = np.reshape(Y_pred, (len(Y_pred), ))

Y_test = np.reshape(Y_test, (len(Y_test), 1))
Y_test = sc_y.inverse_transform(Y_test)
Y_test = np.reshape(Y_test, (len(Y_test), ))

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
print(np.corrcoef(Y_pred, Y_test))
print(r2_score(Y_test, Y_pred))
print(mean_absolute_error(Y_test, Y_pred))


#Y_test = Y_test.values

cnt = 0
for i in range(len(Y_test)):
    if Y_test[i]*Y_pred[i] >=0:
        cnt += 1
        
print(cnt/len(Y_test)*100)







