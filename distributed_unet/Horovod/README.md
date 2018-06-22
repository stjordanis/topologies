# UNet with Horovod (README under construction)

## Overview

## Setup

## Required Data

Data files are not included in this public repo but can accessed by registering (using your institutional email address) at the following link: http://braintumorsegmentation.org/. Once access has been granted, you may download the raw data. To convert those datasets into numpy arrays having shape [num_images, x_dimension (128), y_dimension (128), num_channels] run `python converter.py` after changing its root_dir variable to point to the location your MICCAI_BraTS... folder (processing will take a few minutes). Once complete, the following four files will be saved in the directory specified by the OUT_PATH variable in `settings.py`.

```
-rw-r----- 1 bduser bduser  3250585696 Nov 14 11:42 imgs_test.npy
-rw-r----- 1 bduser bduser 13002342496 Nov 14 12:08 imgs_train.npy
-rw-r----- 1 bduser bduser   406323296 Nov 14 11:36 msks_test.npy
-rw-r----- 1 bduser bduser  1625292896 Nov 14 11:47 msks_train.npy
```

Copy these files to that same directory on all nodes.

## Modifications to settings file

Once environments are constructed which meets the above requirements, clone this repo anywhere on the master node.

Within the cloned directory, open `hosts.txt` and replace the current addresses with the appropriate addresses for your cluster.

Depending on your hardware, you may need to modify the NUM_INTRA_THREADS value. This code was developed on Intel KNL and SKL servers having 68 and 56 cores each, so intra-op thread values of 57 and 50 were most ideal. Please note that maxing out the NUM_INTRA_THREADS value may result in segmentation faults or other memory issues. It is recommended to turn off hyper-threading to prevent resource exhaustion.

Note that a natural consequence of synchronizing updates across several workers is a proportional decrease in the number of weight updates per epoch and slower convergence. To combat this slowdown and reduce the training time in multi-node execution, we default to a large initial learning rate which decays as the model trains. This learning rate is also contained in `settings.py`.

## Multi-Node Execution

## Inference

## Sample Generation/Validation

## Citations

Whenever using and/or refering to the BraTS datasets in your publications, please make sure to cite the following papers.

1. https://www.ncbi.nlm.nih.gov/pubmed/25494501
2. https://www.ncbi.nlm.nih.gov/pubmed/28872634
