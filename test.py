import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, max_error
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import categorical_embedder as ce
#import lightgbm as lgb
import joblib
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor



x_test = pd.read_pickle('./x_test.pkl')
y_test = pd.read_pickle('./y_test.pkl')
model = joblib.load('./model_trained.pkl')


#Predictions which are saved to pickle files for later plotting purposes
y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred)
print(y_pred.info())
y_pred.to_pickle('./test_prediction.pkl') #saving predictions to pickle files for later plotting purposes

#R2
print(r2_score(y_test, y_pred), 'r2 test')

#Range of values in ev
range_test = float(np.max(y_test) - np.min(y_test))

#RMSE
print(np.sqrt(mean_squared_error(y_test, y_pred)), "test RMSE")

#MAE
print(mean_absolute_error(y_test, y_pred), "test MAE")

#MDAE
print(median_absolute_error(y_test, y_pred), "test MDAE")

#MAX_ERROR
print(max_error(y_test, y_pred), "test MAX_ERROR")

#Plotting SI Figure 1
diff = y_test.values - y_pred.values
diff = pd.DataFrame(diff)
print(diff.info())
print(diff.describe())
_ = plt.hist(diff.dropna().values, bins=40)
plt.title('Histogram of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

#Evaulating per adsorbate, here we are only using the test set

ad_list = ['CH2', 'CH3', 'CH', 'C', 'H2O', 'H', 'NH', 'N', 'OH', 'O', 'SH', 'S']
ad_num = [10.396, 9.84, 10.64, 11.2603, 12.65, 13.59844, 12.8, 14.53414, 13.017, 13.61806, 10.4219, 10.36001]
#
for i in range(len(ad_list)):
    x_split = x_test[x_test.ads_IE_1 == ad_num[i]]
    y_split = y_test[x_test.ads_IE_1 == ad_num[i]]
    y_pred_s = model.predict(x_split)
    y_pred_s = pd.DataFrame(y_pred_s)
    print(r2_score(y_split, y_pred_s), 'r2 test', ad_list[i])
    filen =  str(ad_list[i]) + 'pred.pkl'
    y_pred_s.to_pickle(filen)
    print(np.sqrt(mean_squared_error(y_split, y_pred_s)), "test RMSE", ad_list[i])
    print((mean_absolute_error(y_split, y_pred_s)), "test MAE", ad_list[i])
    print((median_absolute_error(y_split, y_pred_s)), "test MDAE", ad_list[i])
    print((max_error(y_split, y_pred_s)), "test MAX_ERROR", ad_list[i])

#Can be used to plot feature importance internally from Catboost
#importance = pd.DataFrame({'feature_importance': model.get_feature_importance(),
#              'feature_names': x_train.columns}).sort_values(by=['feature_importance'],
#                                                       ascending=False)
#importance.to_csv('./imp.csv')
#importance[:20].sort_values(by=['feature_importance'], ascending=True).plot.barh(x='feature_names', y='feature_importance')
#plt.show()

