#!/bin/bash

pip install memory_profiler
rm mprofile*.dat

for dim_length in 32 56 64 80 128 184 256 320 400 512 584
do

   if [ $dim_length -gt 256 ]; then
      num=4
   elif [ $dim_length -gt 100 ]; then
      num=12
   elif [ $dim_length -gt 50 ]; then
      num=200
   else
      num=300
   fi

   # Training batch size 1
   mprof run benchmark_model.py --dim_length $dim_length --num_datapoints $num --epochs 1 --bz 1 \
         2>&1 | tee train_unet_${dim_length}_bz1.log
   mv mprofile*.dat unet3d_train_len${dim_length}_bz1.dat

   sleep 10
   sudo sync; echo 3 > /proc/sys/vm/drop_caches
   sleep 60

   # Training batch size 2
   mprof run benchmark_model.py --dim_length $dim_length --num_datapoints $num --epochs 1 --bz 2 \
         2>&1 | tee train_unet_${dim_length}_bz2.log
   mv mprofile*.dat unet3d_train_len${dim_length}_bz2.dat

   sleep 10
   sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
   sleep 50

   # Inference batch size 1
   mprof run benchmark_model.py --dim_length $dim_length --inference --num_datapoints $num --epochs 1 --bz 1 \
         2>&1 | tee inference_unet_${dim_length}_bz1.log
   mv mprofile*.dat unet3d_inference_len${dim_length}_bz1.dat

   sleep 10
   sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
   sleep 50

done

clear
echo "All done. Remember to copy the logs."
