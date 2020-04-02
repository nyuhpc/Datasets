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

######################### sqlite file - for lmdb and other, excpet hdf5
with sqlite3.connect("TextImages-JPG-10000.sqlite") as sql_conn:
    df = pd.read_sql_query("select * from meta;", sql_conn)
    all_keys = df['key'].tolist()

######################## hdf5 sqlite file
with sqlite3.connect("TextImages-hdf5-10000.sqlite") as sql_conn: 
    df_hdf5 = pd.read_sql_query("select * from meta;", sql_conn)
    all_keys_hdf5 = df_hdf5['key'].tolist()

#############################################################
#############################################################
timing_dict = {"lmdb": [], "hdf5": [], "scratch": [], "SLURM_TMPDIR": [], "SLURM_RAM_TMPDIR": []}

N_to_read = [100, 500, 1000, 3000, 6000, 8000, len(all_keys)]
timing_dict["N_to_read"] = N_to_read    

#for N_of_files in [len(all_keys)]:    
for N_of_files in N_to_read:
    
    ## Sequentital read (next block of lines)
    key_list = all_keys[:N_of_files]
    key_list_hdf5 = all_keys_hdf5[:N_of_files]    
    
    ## Random access (next block of lines)
    #chosen_items = random.sample(range(len(all_keys)), N_of_files)
    #key_list = [all_keys[k] for k in chosen_items]
    #key_list_hdf5 = [all_keys_hdf5[k] for k in chosen_items]
    
    # monitor
    print("first five keys now: ")
    print(key_list[0:5])

    ##############################################################
    print("read data from lmdb")
    env = lmdb.open("TextImages-JPG-10000.lmdb")
    
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
    
    with h5py.File('TextImages-JPG-10000.hdf5', 'r') as f:
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
    
    
    ##############################################################
    print('read data from scratch')
    
    im_ar3 = {}
    tic = time.time()
    
    for key in key_list:
        #print("workign with key: " + key)
        path_to_file = "/scratch/ss13638/datasets/TextRec/data_10000/" + \
                       "mnt/ramdisk/max/90kDICT32px/" + \
                       df[df.key == key]["path"].tolist()[0]
        PIL_image = Image.open(path_to_file) 
        im_ar3[key] = np.asarray(PIL_image)
    
    toc = time.time()
    timing_dict["scratch"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))
    
    ##############################################################
    print("read data from $SLURM_TMPDIR")
    
    im_ar4 = {}
    tic = time.time()
    ## copy files
    tic_1 = time.time()
    os.system("cp -r data_10000/* $SLURM_TMPDIR/")
    print("copy process took (included in total bellow): " + str(time.time() - tic_1))
    
    for key in key_list:
        #print("workign with key: " + key)
        path_to_file = os.environ['SLURM_TMPDIR'] + "/" + \
                       "mnt/ramdisk/max/90kDICT32px/" + \
                       df[df.key == key]["path"].tolist()[0]
        PIL_image = Image.open(path_to_file) 
        im_ar4[key] = np.asarray(PIL_image)
    
    toc = time.time()
    timing_dict["SLURM_TMPDIR"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))
    
    ##############################################################
    print('read data from $SLURM_RAM_TMPDIR')
    
    im_ar5 = {}
    tic = time.time()
    ## copy files
    tic_1 = time.time()
    os.system("cp -r data_10000/* $SLURM_RAM_TMPDIR/")
    print("copy process took (included in total bellow): " + str(time.time() - tic_1))
    
    
    
    for key in key_list:
        #print("workign with key: " + key)
        path_to_file = os.environ['SLURM_RAM_TMPDIR'] + "/" + \
                       "mnt/ramdisk/max/90kDICT32px/" + \
                       df[df.key == key]["path"].tolist()[0]
        PIL_image = Image.open(path_to_file) 
        im_ar5[key] = np.asarray(PIL_image)
    
    toc = time.time()
    timing_dict["SLURM_RAM_TMPDIR"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))
    
print(timing_dict)


########### Plot

N_array = timing_dict["N_to_read"]
del timing_dict["N_to_read"]

df = pd.DataFrame(timing_dict, index=N_array)
df = df.drop(100)
df = df.round(3)

lines = df.plot.line(style='.-', markersize = 20)
lines.set_xlabel("Number of images");
lines.set_ylabel("Time for reading (s)");
plt.savefig('./read_sequential.png')
#plt.savefig('./read_rand.png')


