import pandas as pd
import matplotlib.pyplot as plt
import joblib
import matplotlib

y_test = pd.read_pickle('./y_test.pkl')  
x_test = pd.read_pickle('./x_test.pkl')

y_pred_CH2 = pd.read_pickle('./CH2pred.pkl')
y_pred_CH3 = pd.read_pickle('./CH3pred.pkl')
y_pred_CH = pd.read_pickle('./CHpred.pkl')
y_pred_C = pd.read_pickle('./Cpred.pkl')
y_pred_H2O = pd.read_pickle('./H2Opred.pkl')
y_pred_H = pd.read_pickle('./Hpred.pkl')
y_pred_NH = pd.read_pickle('./NHpred.pkl')
y_pred_N = pd.read_pickle('./Npred.pkl')
y_pred_OH = pd.read_pickle('./OHpred.pkl')
y_pred_O = pd.read_pickle('./Opred.pkl')
y_pred_SH = pd.read_pickle('./SHpred.pkl')
y_pred_S = pd.read_pickle('./Spred.pkl')


ad_list = ['CH2', 'CH3', 'CH', 'C', 'H2O', 'H', 'NH', 'N', 'OH', 'O', 'SH', 'S']
ad_num = [10.396, 9.84, 10.64, 11.2603, 12.65, 13.59844, 12.8, 14.53414, 13.017, 13.61806, 10.4219, 10.36001]

y_test_CH2 = y_test[x_test.ads_IE_1 == 10.396]
y_test_CH3 = y_test[x_test.ads_IE_1 == 9.84]
y_test_CH = y_test[x_test.ads_IE_1 == 10.64]
y_test_C = y_test[x_test.ads_IE_1 == 11.2603]
y_test_H2O = y_test[x_test.ads_IE_1 == 12.65]
y_test_H = y_test[x_test.ads_IE_1 == 13.59844]
y_test_NH = y_test[x_test.ads_IE_1 == 12.8]
y_test_N = y_test[x_test.ads_IE_1 == 14.53414]
y_test_OH = y_test[x_test.ads_IE_1 == 13.017]
y_test_O = y_test[x_test.ads_IE_1 == 13.61806]
y_test_SH = y_test[x_test.ads_IE_1 == 10.4219]
y_test_S = y_test[x_test.ads_IE_1 == 10.36001]

matplotlib.rcParams.update({'font.size': 14})
params = {'mathtext.default': 'regular' }          
matplotlib.rcParams.update(params)

fig, (ax1, ax2) = plt.subplots(ncols=2)
#Polyatomic
ax1.scatter(y_test_CH2, y_pred_CH2, s=12, c='tab:blue', marker="o", label='$CH_2$; $r^2$=0.920')
ax1.scatter(y_test_CH3, y_pred_CH3, s=12, c='tab:orange', marker="v", label='$CH_3$; $r^2$=0.901')
ax1.scatter(y_test_CH, y_pred_CH, s=12, c='tab:green', marker="^", label='CH; $r^2$=0.934')
ax1.scatter(y_test_H2O, y_pred_H2O, s=12, c='tab:red', marker="8", label='$H_2$O; $r^2$=0.715')
ax1.scatter(y_test_NH, y_pred_NH, s=12, c='tab:purple', marker="s", label='NH; $r^2$=0.909')
ax1.scatter(y_test_OH, y_pred_OH, s=12, c='tab:brown', marker="*", label='OH; $r^2$=0.964')
ax1.scatter(y_test_SH, y_pred_SH, s=12, c='tab:pink', marker="P", label='SH; $r^2$=0.863')
ax1.legend(loc='upper left')

#Monoatomic
ax2.scatter(y_test_C, y_pred_C, s=12, c='tab:blue', marker="o", label='C; $r^2$=0.999')
ax2.scatter(y_test_H, y_pred_H, s=12, c='tab:orange', marker="v", label='H; $r^2$=0.999')
ax2.scatter(y_test_N, y_pred_N, s=12, c='tab:green', marker="^", label='N; $r^2$=0.998')
ax2.scatter(y_test_O, y_pred_O, s=12, c='tab:red', marker="8", label='O; $r^2$=0.996')
ax2.scatter(y_test_S, y_pred_S, s=12, c='tab:purple', marker="s", label='S; $r^2$=0.974')
ax2.legend(loc='upper left')


# Set common labels
fig.text(0.5, 0.04, 'Test (kcal/mol)', ha='center', va='center')
fig.text(0.04, 0.5, 'Prediction (kcal/mol)', ha='center', va='center', rotation='vertical')

ax1.set_title('Polyatomic Ads.')
ax2.set_title('Monoatomic Ads.')
plt.show()
    
