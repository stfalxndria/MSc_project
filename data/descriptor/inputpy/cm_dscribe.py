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

#running dscribe 

cm_dscribe = []
cm_time = []
cm_ds = CoulombMatrix(n_atoms_max=54,permutation="eigenspectrum")
start = datetime.now()
for i,ase_struct in enumerate(ase_structures):
    dscribe_matrix = cm_ds.create(ase_struct)
    cm_dscribe.append(dscribe_matrix)
    now = datetime.now()
    if (i+1)%200 == 0: 
        cm_time.append("matrices read:", len(cm_dscribe),", time:", (now - start))
   
cm_time.append('Number of matrices read: ', len(cm_dscribe))
cm_time.append("--- %s time taken ---" % (datetime.now()- start))

cm_dscribe = pd.DataFrame(cm_dscribe)
cm_dscribe.to_pickle('cm_dscribe.pkl')

