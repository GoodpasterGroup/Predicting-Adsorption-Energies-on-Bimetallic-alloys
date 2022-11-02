import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import categorical_embedder as ce


d = pd.read_csv('./full_dataset.csv')
xx = d[['surface_composition', 'coverages', 'equation', 'ads_1', 'site_1', 'site_2', 'ads_2', 'site_3', 'ads_3', 'band_gap','efermi','formation_energy_per_atom','total_magnetization','volume','energy_per_atom', 'ads_IE_1', 'ads_H_1','ads_S_1', 'ads_IE_2','ads_H_2','ads_S_2','ads_IE_3','ads_H_3','ads_S_3']]
yy = d['reaction_energy'] * 23.0609 #to kcal/mol

#preprocess
embedding_info = ce.get_embedding_info(xx)

X_encoded,encoders = ce.get_label_encoded_data(xx)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_encoded,yy, test_size=0.1)

embeddings = ce.get_embeddings(X_train_c, y_train_c, categorical_embedding_info=embedding_info,
                            is_classification=False, epochs=1000)
x_train = ce.fit_transform(X_train_c, embeddings=embeddings, encoders=encoders, drop_categorical_vars=True)
x_test = ce.fit_transform(X_test_c, embeddings=embeddings, encoders=encoders, drop_categorical_vars=True)
y_train = y_train_c
y_test = y_test_c

print(len(x_train))
print(len(y_train))

#test train split

x_train = pd.DataFrame(x_train)
print(x_train.info(verbose=True))
print(x_train.describe())
print(x_train.sample())
y_train = pd.DataFrame(y_train)
x_test = pd.DataFrame(x_test)
y_test = pd.DataFrame(y_test)

x_train.to_pickle('./end_2/x_train.pkl')
x_test.to_pickle('./end_2/x_test.pkl')
y_train.to_pickle('./end_2/y_train.pkl')
y_test.to_pickle('./end_2/y_test.pkl')
xx.to_pickle('./end_2/x.pkl')
yy.to_pickle('./end_2/y.pkl')  
