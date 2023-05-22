#importing all of the files
import sys
sys.path.append('/home/uccaset/miniconda3/envs/mlenv/bin')
import os
import copy
import json
import itertools
from pathlib import Path
import shutil as sh
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from CRYSTALpytools.crystal_io import Crystal_output, Crystal_input, Crystal_density, Crystal_gui
from CRYSTALpytools.convert import cry_gui2pmg, cry_out2pmg
from CRYSTALpytools.utils import view_pmg

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, PointGroupAnalyzer

from ase.visualize import view

from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import EwaldSumMatrix

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, max_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,15)


#reading the files
structures = pd.read_pickle('structures.pkl')
structures = structures.to_numpy()
structures = structures.tolist()

ase_structures = pd.read_pickle('ase_structures.pkl')
ase_structures = ase_structures.to_numpy()
ase_structures = ase_structures.tolist()

mbtr_dscribe = pd.read_pickle('./descriptor/mbtr_dscribe.pkl')
mbtr_dscribe = mbtr_dscribe.to_numpy()
mbtr_dscribe = mbtr_dscribe.tolist()

ener_ds = pd.read_pickle('ener_ds.pkl')
energies_sp = ener_ds['energies_sp']
energies_sp = energies_sp.to_numpy()
gap_sp = ener_ds['gap_sp']


#importing all of the functions
def r2(real,pred):
    r2 = r2_score(real, pred)
    return r2

def mae(real,pred):
    mae = mean_absolute_error(real, pred)*1000
    return mae

#maximum error
def maxer(real,pred):
    maxer = max_error(real, pred)*1000
    return maxer

def errorgraph(real,pred, descriptor, model):
    mae(real, pred)
    r2(real, pred)
    maxer(real,pred)
    
    plt.figure(dpi=200)
    plt.scatter(real, pred, marker='o')
    plt.ylabel("Predicted values")
    plt.xlabel("Calculated values")
    
    vmin=min(min(real),min(pred))
    vmax=max(max(real),max(pred))
    line=np.linspace(vmin,vmax)
    plt.plot(line,line,color='green')
    
    
    plt.title('%s %s' %(descriptor, model), fontsize=16)
    
    plt.show()
    plt.close()
    print('r squared value is', r2(real,pred))
    print('mean absolute error', mae(real,pred))
    print('maximum error', maxer(real,pred))

def trainsize(descriptor, energies, model): 
    test_para = np.arange(0.1,1,0.1)
    r2_ = []
    mae_ = []
    maxer_ = []
    paratesting = pd.DataFrame()
    paratesting['test size'] = test_para
    for i in test_para:
        X_train, X_test, y_train, y_test = train_test_split(descriptor, energies, random_state=1, test_size = i)
        scaler = StandardScaler()  
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test)
            
        model.fit(X_train, y_train)
        ypred_LR = model.predict(X_test)

        r2_.append(r2(y_test,ypred_LR))
        mae_.append(mae(y_test, ypred_LR))
        maxer_.append(maxer(y_test,ypred_LR))
            
    paratesting['r^2 value'] = r2_
    paratesting['mean absolute error'] = mae_
    paratesting['maximum error'] = maxer_
    
    return paratesting

def sizeplot(parafile, parameter):
    r2_ = parafile['r^2 value'].tolist()
    mae_ = parafile['mean absolute error'].tolist()
    maxer_ = parafile['maximum error'].tolist()
    nn = parameter.tolist()
    
    
    fig, ax1 = plt.subplots(dpi=200)
    ax2 = ax1.twinx()
    
    ax1.plot(nn, r2_, marker='x', label='r^2 value')
    ax2.plot(nn, mae_, marker='o',color='green',label='mean absolute error')
    #ax2.plot(nn, maxer_, marker='o',color='red', label='maximum error')
    
    ax1.set_xlabel('tested parameter')
    ax1.set_ylabel('r^2 value')
    ax2.set_ylabel('error/ meV')
    
    
    plt.show()
    plt.close()

#plotting ofstandard deviation for each elements in a descriptor:
def std_plot(descriptor):
    des = pd.DataFrame(descriptor)
    std_ = []
    for i in range(len(des.iloc[0])):
        elem = des[i].to_numpy()
        std_.append(np.std(elem))
    return std_

from sklearn.linear_model import LinearRegression
model = LinearRegression()

descriptor = mbtr_dscribe
energies = energies_sp
mbtr_lr = trainsize(descriptor, energies, model)

X_train, X_test, y_train, y_test = train_test_split(descriptor, energies, random_state=1, test_size = 0.3)

scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

model.fit(X_train, y_train)
ypred_LR = model.predict(X_test)

errorgraph(y_test, ypred_LR, 'mbtr', 'linear regression')

mbtr_lr = mbtr_lr.to_pickle('mbtr_lr.pkl')
