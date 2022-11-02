import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

#read in data

x_train = pd.read_pickle('./x_train.pkl')
y_train = pd.read_pickle('./y_train.pkl') 

#Fit model

model = CatBoostRegressor(iterations=15000, learning_rate=0.1, objective='RMSE', depth=8, bootstrap_type='Bernoulli', subsample=1.0, sampling_frequency='PerTree', langevin=True, diffusion_temperature=20000, leaf_estimation_iterations=2, leaf_estimation_backtracking='AnyImprovement')
_ = model.fit(x_train, y_train)

#save model
joblib.dump(model, 'model_trained.pkl')
