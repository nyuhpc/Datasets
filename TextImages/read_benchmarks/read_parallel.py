#srun --ntasks-per-node=16 --nodes 1 --mem=20GB -t1:00:00 --pty /bin/bash

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
import math


######################################################  Correct affinity
## see current affinity 
import os; os.sched_getaffinity(0)
# reset affinity - read about affinity in man taskset
os.system("taskset -p 0xFFFFFFFF %d" % os.getpid())
# check
os.sched_getaffinity(0)

#############################################################
#############################################################
timing_dict = {"lmdb": [], "hdf5": []}

## Assuming we have 16 cpus avaialble on node
## Do 16 codes two times - if any caching is happeing it will happen for first 16 cores run, and the rest of benchmark will happen on already cached data
cpus_options = [16, 16, 8, 4, 2, 1]
timing_dict["cpus_options"] = cpus_options

## Read fixed number of images for all experiments
N_to_read = int(1e5)

########################################################### LMDB

key_list = [f"{i:07}" for i in list(range(1, N_to_read+1))]
lmdb_path = "/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/lmdb/TextImages.lmdb"

## we open env in every process. Opening it for every key is slow - thus pass chunk of keys 
def read_fig_lmdb(key_sublist):
  ret_ar = []
  #print("process id:" + str(os.getpid()))
  with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
    with env.begin() as lmdb_txn:
      for key in key_sublist:
        #print(key)
        stored_image = lmdb_txn.get(key.encode())
        PIL_image = Image.open(io.BytesIO(stored_image))
        ret_ar.append(np.asarray(PIL_image))
  return(ret_ar)


for cpus_number in cpus_options:
    ## split key_list to chunks
    n = math.ceil(len(key_list)/cpus_number)
    key_list_chunks = [key_list[i:i + n] for i in range(0, len(key_list), n)]     
    tic = time.time()
    res = Parallel(n_jobs=cpus_number, prefer = "processes", verbose=5)(delayed(read_fig_lmdb)(key_sublist) for key_sublist in key_list_chunks)
    toc = time.time()
    timing_dict["lmdb"].append(toc-tic)

flattened = [val for sublist in res for val in sublist]
#flattened

    
######################################################### HDF5
hdf5_path = '/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/hdf5/TextImages.hdf5'

def read_fig_hdf5(key_sublist):
  ret_ar = []
  with h5py.File(hdf5_path, 'r') as f:
    for key in key_sublist:
      dset = f['images']
      stored_image = dset[key]
      #print(data)
      PIL_image = Image.open(io.BytesIO(stored_image))
      ret_ar.append(np.asarray(PIL_image))
  return(ret_ar)


key_list_hdf5 = list(range(int(N_to_read)))
for cpus_number in cpus_options:
    ## split key_list to chunks
    n = math.ceil(len(key_list_hdf5)/cpus_number)
    key_list_chunks = [key_list_hdf5[i:i + n] for i in range(0, len(key_list_hdf5), n)]     
    tic = time.time()
    res = Parallel(n_jobs=cpus_number, prefer = "processes", verbose=5)(delayed(read_fig_hdf5)(key_sublist) for key_sublist in key_list_chunks)
    toc = time.time()
    timing_dict["hdf5"].append(toc-tic)

flattened = [val for sublist in res for val in sublist]


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


