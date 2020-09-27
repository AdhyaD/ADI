import xlrd
import pandas as pd
import sklearn
import math

from sklearn import metrics

df = pd.read_excel('/Users/adhyadagar/Desktop/ADI_RMSE.xlsx')


actual_2011 = df['ADI_2011']
ma_max_2011 = actual_2011.max()
ma_min_2011 = actual_2011.min()
predicted_2011 = df['PREDICTED_ADI_2011']
mp_2011 = predicted_2011.mean()
diff_2011 = ma_max_2011 - ma_min_2011

actual_2001 = df['ADI_2001']
predicted_2001 = df['PREDICTED_ADI_2001']
ma_max_2001 = actual_2001.max()
ma_min_2001 = actual_2001.min()
mp_2001 = predicted_2001.mean()
diff_2001 = ma_max_2001 - ma_min_2001

mse_2011 = sklearn.metrics.mean_squared_error(actual_2011, predicted_2011)
rmse_2011 = math.sqrt(mse_2011)
nrmse_2011 = rmse_2011/diff_2011

mse_2001 = sklearn.metrics.mean_squared_error(actual_2001, predicted_2001)
rmse_2001 = math.sqrt(mse_2001)
nrmse_2001 = rmse_2011/diff_2001


print("rmse_2011",rmse_2011)
print("nrmse_2011",nrmse_2011)
print('######')

print("rmse_2001",rmse_2001)
print("nrmse_2001",nrmse_2001)
