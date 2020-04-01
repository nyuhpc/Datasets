Procedure:

* prepare
  + create_subset_of_files.sh
  + run write_jpg_hdf5_10000.py
  + run write_jpg_lmdb_10000.py
* read test
  + adjust file read_test.py as needed: comment/uncomment (## Sequentital read) or (## Random access). Same change shall be done to line saving plot in the end
  + run batch job with slurm_send file