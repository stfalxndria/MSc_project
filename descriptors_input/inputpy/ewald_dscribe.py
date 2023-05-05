#importing all of the files
import os
import copy
import json
import itertools
import shutil as sh
from pathlib import Path
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

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,15)


#reading the files
structures = pd.read_pickle('structures.pkl')
structures = structures.to_numpy()
structures = structures.tolist()

ase_structures = pd.read_pickle('ase_structures.pkl')
ase_structures = ase_structures.to_numpy()
ase_structures = ase_structures.tolist()

ener_ds = pd.read_pickle('ener_ds.pkl')

ewald = EwaldSumMatrix(
    n_atoms_max=54,
)

ewald_dscribe = []
ewald_time = []
start = datetime.now()
for i,ase_struct in enumerate(ase_structures):
    ewald_matrix = ewald.create(ase_struct)
    ewald_dscribe.append(ewald_matrix)
    now = datetime.now()
    if int(len(ewald_dscribe)) == 20:
        ewald_time.append("matrices read:", len(ewald_dscribe),", time:", (now - start))
    if int(len(ewald_dscribe))%200 == 0: 
        ewald_time.append("matrices read:", len(ewald_dscribe),", time:", (now - start))
ewald_time.append('Number of matrices read: ', len(ewald_dscribe))
ewald_time.append("--- %s time taken ---" % ((datetime.now() - start)))

ewald_dscribe = pd.DataFrame(ewald_dscribe)
ewald_dscribe.to_pickle('ewald_dscribe.pkl')



