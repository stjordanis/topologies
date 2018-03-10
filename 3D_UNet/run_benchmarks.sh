#!/bin/sh

# ----------------------------------------------------------------------------
# Copyright 2018 Intel
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

for dm in 64; do #64 128 256; do
  for bz in 56; do #1 2 4 8; do
    echo -e "\n\n #### Starting size $dm x $dm x $dm and BZ=$bz ####\n\n"
    /usr/bin/time -v python benchmark_model.py --dim_length $dm \
           --bz $bz --num_datapoints=128 \
           2>&1 | tee 3D_unet_${dm}_bz_${bz}.log
    echo -e "#### Finished $dm x $dm x $dm and BZ=$bz ####"
  done
done
