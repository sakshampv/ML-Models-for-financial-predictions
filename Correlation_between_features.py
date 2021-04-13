import pandas as pd
import numpy as np
df_train = pd.read_excel (r'./ML_Data/TrainingData2.xlsx')
X = df_train.drop(columns = [ 'Stock Name', 'Date'], inplace = False)
#X = pd.read_csv("PCA_Projected_Data.csv")
y = df_train['PX_LAST']
X = X.drop(columns = [ "PX_LAST"], inplace = False)

a = X.columns.values

a = np.asarray(a)
a = a.astype('str')

corrs = pd.DataFrame()
corrs['Feature'] = a

q = []

for column in a:
    b = X[column].values
    q += [(np.corrcoef(X[column].values,y.values)[0,1])]


corrs['Correlation with Returns'] = q

corrs = corrs.sort_values(by='Correlation with Returns', ascending=False)

corrs.to_csv("Correlation_bw_return_and_features_sorted.csv")

