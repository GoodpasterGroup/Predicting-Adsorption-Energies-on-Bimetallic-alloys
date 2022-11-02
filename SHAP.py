import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
import pandas as pd
import re
import matplotlib as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
#import category_encoders as ce


x = pd.read_pickle('./x_train.pkl')
x = pd.DataFrame(x)
model = joblib.load('model_trained.pkl')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)
#shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], x.iloc[0,:])
#shap.summary_plot(shap_values, x, max_display=10)
shap.plots.bar(explainer(x), max_display=11)
plt.show()
    
