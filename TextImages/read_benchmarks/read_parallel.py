#srun --ntasks-per-node=16 --nodes 1 --mem=10GB -t1:00:00 --pty /bin/bash

from joblib import Parallel, delayed, parallel_backend
import numpy as np
from PIL import Image
import lmdb
from random import sample
import sqlite3
import pandas as pd
import io
import time
import os
import random
import h5py
import matplotlib.pyplot as plt



######################################################  Correct affinity
## see current affinity 
import os; os.sched_getaffinity(0)
# reset affinity - read about affinity in man taskset
os.system("taskset -p 0xFFFFFFFF %d" % os.getpid())
# check
os.sched_getaffinity(0)

#############################################################
read sqlite files and get all keys
print("---reading sqlite for lmdb")
######################## sqlite file - for lmdb
with sqlite3.connect("/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/lmdb/TextImages.sqlite") as sql_conn:
   df = pd.read_sql_query("select * from meta;", sql_conn)
   all_keys = df['key'].tolist()

## let say we want to get first image
# key = all_keys[0]

print("---reading sqlite for hdf5")
####################### hdf5 sqlite file
with sqlite3.connect("/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/hdf5/TextImages-hdf5.sqlite") as sql_conn:
   df_hdf5 = pd.read_sql_query("select * from meta;", sql_conn)
   all_keys_hdf5 = df_hdf5['key'].tolist()

#############################################################
#############################################################
timing_dict = {"lmdb": [], "hdf5": []}

## Assuming we have 16 cpus avaialble on node
#cpus_options = [1, 2, 4, 8, 16]
cpus_options = [16, 16, 8, 4, 2, 1]
timing_dict["cpus_options"] = cpus_options

## Read fixed number of images for all experiments
N_to_read = int(1e5)

########################################################### LMDB

key_list = [f"{i:07}" for i in list(range(1, N_to_read+1))]
lmdb_path = "/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/lmdb/TextImages.lmdb"

def read_fig_lmdb(key):
  #print("process id:" + str(os.getpid()))
  with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
    with env.begin() as lmdb_txn:
      #print(key)
      stored_image = lmdb_txn.get(key.encode())
      #print(data)
      PIL_image = Image.open(io.BytesIO(stored_image))
      return(np.asarray(PIL_image))


for cpus_number in cpus_options:
    tic = time.time()
    res = Parallel(n_jobs=cpus_number, prefer = "processes", verbose=5)(delayed(read_fig_lmdb)(key) for key in key_list)
    toc = time.time()
    timing_dict["lmdb"].append(toc-tic)
    
######################################################### HDF5

def read_fig_hdf5(key):
    with h5py.File('/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/hdf5/TextImages.hdf5', 'r') as f:
        dset = f['images']
        stored_image = dset[key]
        #print(data)
        PIL_image = Image.open(io.BytesIO(stored_image))
        return(np.asarray(PIL_image))


key_list = list(range(int(N_to_read)))
for cpus_number in cpus_options:
    tic = time.time()
    res = Parallel(n_jobs=cpus_number, prefer = "processes", verbose=5)(delayed(read_fig_hdf5)(key) for key in key_list)
    toc = time.time()
    timing_dict["hdf5"].append(toc-tic)

#######################################################
import sys
print("size or loaded data: " + str(sys.getsizeof(res)))
## size or loaded data: 1730440
## overhead ~100MB per process

print(timing_dict)    

######################################################

########### Plot
cpus_options = timing_dict["cpus_options"]
del timing_dict["cpus_options"]

df = pd.DataFrame(timing_dict, index=cpus_options)
df = df.round(3)

lines = df.plot.line(style='.-', markersize = 20)
lines.set_title("Reading time depending on number of cores used")
lines.set_xlabel("Number of cores used");
lines.set_ylabel("Time for reading (s)");
plt.savefig('./read_parallel.png')

#lines.set_xscale('log')
lines.set_yscale('log')
plt.savefig('./read_parallel_log.png')


