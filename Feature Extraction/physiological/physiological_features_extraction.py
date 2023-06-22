import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

import os
from pathlib import Path

import pylab 
import scipy.stats as stats
import neurokit2 as nk

from ecg_features import *
from eda_features import *
from respiration_features import *


####### LOAD DATA
path_data = "../../../Dataset/Physiological/"

filelist= [f for f_sub in [f_names for root,d_names,f_names in os.walk(path_data)] for f in f_sub]
dirlist= [d for d_sub in [d_names for root,d_names,f_names in os.walk(path_data)] for d in d_sub]

data_ecg = dict()
data_eda = dict()
data_rsp = dict()

filelist.sort()

for i in filelist:
    if not i.startswith('.'):
        path = os.path.join(path_data,i.split('_')[0]+'/')
        file = pd.read_csv(path+i, sep=",")
        if file.isnull().sum().sum() != 0:
            print('There are ', file.isnull().sum().sum(), ' nan values in the recording', i)

        file_ecg = np.array(file['ECG'])
        file_eda = np.array(file['EDA'])
        file_rsp = np.array(file['RR'])

        data_ecg[i.split('.')[0]] = (file_ecg - file_ecg.mean())/file_ecg.std()
        data_eda[i.split('.')[0]] = (file_eda - file_eda.mean())/file_eda.std()
        data_rsp[i.split('.')[0]] = (file_rsp - file_rsp.mean())/file_rsp.std()
        
        
del data_eda['r5s8_Counting3'] 


####### CLEAN USING NK
ecg_clean = data_ecg.copy()
eda_clean = data_eda.copy()
rsp_clean = data_rsp.copy()

for ecg,eda,rsp in zip(data_ecg.items(), data_eda.items(), data_rsp.items()):
    ecg_clean[ecg[0]] = nk.ecg_clean(ecg[1], sampling_rate=500, method="biosppy")
    eda_clean[eda[0]] = nk.eda_clean(eda[1], sampling_rate=500,method='biosppy')
    rsp_clean[rsp[0]] = nk.rsp_clean(rsp[1], sampling_rate=500, method="biosppy")

    
    
######## EXTRACT FEATURES BY MODALITY    
df_eda_features = get_eda_features(eda_clean, 500)
print('EDA: {0:2d} trials and {1:2d} features'.format(df_eda_features.shape[0], df_eda_features.shape[1]))

df_rsp_features = get_resp_features(rsp_clean, 500)
print('Respiration : {0:2d} trials and {1:2d} features'.format(df_rsp_features.shape[0], df_rsp_features.shape[1]))

df_ecg_features = get_ecg_features(ecg_clean, 500)
print('ECG : {0:2d} trials and {1:2d} features'.format(df_ecg_features.shape[0], df_ecg_features.shape[1]))


######## MERGE
df_features = pd.concat([df_ecg_features, df_rsp_features], axis=1).merge(df_eda_features, left_index= True, right_index=True)




####### EXPORT
df_eda_features.to_csv('../Features/eda_features.csv', sep=",", index=True)
df_rsp_features.to_csv('../Features/resp_features.csv', sep=",", index=True)
df_ecg_features.to_csv('../Features/ecg_features.csv', sep=",", index=True)
df_features.to_csv('../Features/all_physiological_features.csv', sep=",", index=True)


        
