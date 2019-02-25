#!/bin/bash

#pip install memory_profiler
rm *.dat
rm *.log

CMDS="--epochs 3 --ngraph"

for dim_length in 200 #256 #32 56 64 80 128 184 200 #256 320 400 480 512 600
do

   num=1000000  # Run for a long time because timeout will automatically stop the script after a certain number of seconds
   secs=4000  # Number of seconds to record memory

   # Training batch size 1
   echo "Training batch size 1, dim_length ${dim_length}"
   timeout $secs mprof run \
                 --output unet3d_train_len${dim_length}_bz1.dat \
                 python benchmark_model.py \
                 --dim_lengthx $dim_length \
                 --dim_lengthy $dim_length \
                 --dim_lengthz $dim_length \
                 --num_datapoints $num --bz 1 $CMDS \
                 2>&1 | tee train_unet_${dim_length}_bz1.log

   bash clear_caches.sh

   # Inference batch size 1
   echo "Inference batch size 1, dim_length ${dim_length}"
   timeout $secs mprof run \
                 --output unet3d_inference_len${dim_length}_bz1.dat \
                 python benchmark_model.py \
                 --dim_lengthx $dim_length \
                 --dim_lengthy $dim_length \
       	       	 --dim_lengthz $dim_length \
                 --num_datapoints $num $CMDS --bz 1 --inference \
                 2>&1 | tee inference_unet_${dim_length}_bz1.log

   bash clear_caches.sh

   # Inference batch size 2
   echo "Inference batch size 2, dim_length ${dim_length}"
   timeout $secs mprof run \
                 --output unet3d_inference_len${dim_length}_bz2.dat \
                 python benchmark_model.py \
                 --dim_lengthx $dim_length \
                 --dim_lengthy $dim_length \
       	       	 --dim_lengthz $dim_length \
                 --num_datapoints $num $CMDS --bz 2 --inference \
                 2>&1 | tee inference_unet_${dim_length}_bz2.log

   bash clear_caches.sh

   # Inference batch size 4
   echo "Inference batch size 4, dim_length ${dim_length}"
   timeout $secs mprof run \
                 --output unet3d_inference_len${dim_length}_bz4.dat \
                 python benchmark_model.py \
                 --dim_lengthx $dim_length \
                 --dim_lengthy $dim_length \
       	       	 --dim_lengthz $dim_length \
                 --num_datapoints $num $CMDS --bz 4 --inference \
                 2>&1 | tee inference_unet_${dim_length}_bz4.log

   bash clear_caches.sh

   # Inference batch size 2
   # echo "Training batch size 2, dim_length ${dim_length}"
   # timeout $secs mprof run \
   #               --output unet3d_train_len${dim_length}_bz2.dat \
   #               python benchmark_model.py \
   #               --dim_lengthx $dim_length \
   #               --dim_lengthy $dim_length \
   #     	       	 --dim_lengthz $dim_length \
   #               --num_datapoints $num $CMDS --bz 2  \
   #               2>&1 | tee inference_unet_${dim_length}_bz2.log
   #
   # bash clear_caches.sh
   #
   # # Inference batch size 4
   # echo "Training batch size 4, dim_length ${dim_length}"
   # timeout $secs mprof run \
   #               --output unet3d_train_len${dim_length}_bz4.dat \
   #               python benchmark_model.py \
   #               --dim_lengthx $dim_length \
   #               --dim_lengthy $dim_length \
   #     	       	 --dim_lengthz $dim_length \
   #               --num_datapoints $num $CMDS --bz 4  \
   #               2>&1 | tee inference_unet_${dim_length}_bz4.log
   #
   # bash clear_caches.sh

done

echo "Done"
