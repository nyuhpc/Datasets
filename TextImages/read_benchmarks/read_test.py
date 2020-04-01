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
##############################################################

# read sqlite files and get all keys

print("---reading sqlite for lmdb")
######################### sqlite file - for lmdb 
with sqlite3.connect("/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/lmdb/TextImages.sqlite") as sql_conn:
    df = pd.read_sql_query("select * from meta;", sql_conn)
    all_keys = df['key'].tolist()

## let say we want to get first image
# key = all_keys[0]

print("---reading sqlite for hdf5")
######################## hdf5 sqlite file
with sqlite3.connect("/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/hdf5/TextImages-hdf5.sqlite") as sql_conn:
    df_hdf5 = pd.read_sql_query("select * from meta;", sql_conn)
    all_keys_hdf5 = df_hdf5['key'].tolist()

#############################################################
#############################################################
timing_dict = {"lmdb": [], "hdf5": []}

N_to_read = [100, 300, 1000, 3000, 10000, 30000, 100000]
timing_dict["N_to_read"] = N_to_read    

print("---reading files")

#for N_of_files in [len(all_keys)]:    
for N_of_files in N_to_read:
    
    ## Sequentital read (next block of lines)
    #key_list = all_keys[:N_of_files]
    #key_list_hdf5 = all_keys_hdf5[:N_of_files]    
    
    ## Random access (next block of lines)
    random.seed(1)
    chosen_items = random.sample(range(len(all_keys)), N_of_files)
    key_list = [all_keys[k] for k in chosen_items]
    key_list_hdf5 = [all_keys_hdf5[k] for k in chosen_items]
    
    # monitor
    print("first five keys now: ")
    print(key_list[0:5])

    ##############################################################
    print("read data from lmdb")
    # make sure you specify following parameters: readonly=True, lock=False
    env = lmdb.open("/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/lmdb/TextImages.lmdb",
                readonly=True, lock=False)
    
    im_ar = {}
    
    tic = time.time()
    
    with env.begin() as lmdb_txn:
        for key in key_list:
            #print("workign with key: " + key)
            stored_image = lmdb_txn.get(key.encode())
            #print(data)
            PIL_image = Image.open(io.BytesIO(stored_image))
            im_ar[key] =  np.asarray(PIL_image)
    env.close()
    toc = time.time()
    timing_dict["lmdb"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))
    
    ##############################################################
    print('read data from hdf5')
    
    im_ar2 = {}
    tic = time.time()
    
    with h5py.File('/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/hdf5/TextImages.hdf5', 'r') as f:
        dset = f['images']
        for key in key_list_hdf5:
            #print("workign with key: " + key)
            stored_image = dset[key]
            #print(data)
            PIL_image = Image.open(io.BytesIO(stored_image))
            im_ar2[key] =  np.asarray(PIL_image)
    
    toc = time.time()
    timing_dict["hdf5"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))
    
    
print(timing_dict)


########### Plot

N_array = timing_dict["N_to_read"]
del timing_dict["N_to_read"]

df = pd.DataFrame(timing_dict, index=N_array)
#df = df.drop(100)
df = df.round(3)

lines = df.plot.line(style='.-', markersize = 20)
lines.set_xlabel("Number of images");
lines.set_ylabel("Time for reading (s)");
#plt.savefig('./read_sequential.png')
plt.savefig('./read_rand.png')


