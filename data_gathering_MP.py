import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

mpdr = MPDataRetrieval()
d = pd.read_csv('./final_5.csv')
df = d['surface_composition'].unique()
df = pd.DataFrame(df, columns=['surface_composition'])
#print(df_2.info())
name_list = df['surface_composition'].tolist()
#with MPRester("EhrOoOM5RpfLj4bxhToB") as m:
df['density'] = np.nan
df['band_gap'] = np.nan
df['efermi'] = np.nan
df['formation_energy_per_atom'] = np.nan
df['total_magnetization'] = np.nan
df['volume'] = np.nan
df['energy_per_atom'] = np.nan
df['pretty_formula'] = np.nan
df['material_id'] = np.nan
for index, row in df.iterrows():
    name = row['surface_composition']
    df2 = mpdr.get_dataframe(criteria=name, properties=['density', 'band_gap','efermi', 'formation_energy_per_atom','total_magnetization','volume','energy_per_atom','pretty_formula', 'material_id'])
    if len(df2) > 1:
       df2 = df2[df2.energy_per_atom == df2.energy_per_atom.min()]
       #print(len(df2))
    #df2 = df2.sort_values(by='energy_per_atom', ascending=True)
    df2 = df2.reset_index()
    if len(df2) > 0:
       df.loc[index, 'density'] = df2['density'][0]
       df.loc[index, 'band_gap'] = df2['band_gap'][0] 
       df.loc[index, 'efermi'] = df2['efermi'][0]
       df.loc[index, 'formation_energy_per_atom'] = df2['formation_energy_per_atom'][0]
       df.loc[index, 'total_magnetization'] = df2['total_magnetization'][0]
       df.loc[index, 'volume'] = df2['volume'][0]
       df.loc[index, 'energy_per_atom'] = df2['energy_per_atom'][0]
       df.loc[index, 'pretty_formula'] = df2['pretty_formula'][0]
       df.loc[index, 'material_id'] = df2['material_id'][0]
df.to_csv('all.csv')

