# 3D U-Net for Medical Decathlon Dataset

![pred152_3D](https://github.com/NervanaSystems/topologies/blob/master/3D_UNet/keras_training_only_version/images/BRATS_152_img3D.gif
"BRATS image #152:  Purple voxels indicate a perfect prediction by the model. Red are false positives. Blue are false negatives").

Trains a 3D U-Net on the brain tumor segmentation ([BraTS](https://www.med.upenn.edu/sbia/brats2017.html)) subset of the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) dataset. This model can achieve Dice coefficient of > 0.85.

Steps to train a new model:

1. Go to the [Medical Segmentation Decathlon](http://medicaldecathlon.com) website and download the [BraTS subset](https://drive.google.com/file/d/1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU/view?usp=sharing). The dataset has the Creative Commons Attribution-ShareAlike 4.0 International [license](https://creativecommons.org/licenses/by-sa/4.0/).
2. Untar the "Task01_BrainTumour.tar" file ```tar -xvf Task01_BrainTumour.tar```
3. Create a Conda environment with TensorFlow. Command: ```conda create -c anaconda -n decathlon pip python=3.6 tensorflow keras tqdm h5py psutil```
4. Enable the new environment. Command: ```conda activate decathlon```
5. Install the package [nibabel](http://nipy.org/nibabel/). Command: `pip install nibabel`
6. Run the command ```python train.py --data_path $DECATHLON_ROOT_DIRECTORY```, where `$DECATHLON_ROOT_DIRECTORY` is the root directory where you un-tarred the Decathlon dataset.


Steps to evaluate a pre-trained 3D U-Net model.

1. Download the [Medical Decathlon dataset](http://medicaldecathlon.com/). Specifically, this model was trained on the brain tumor segmentation (BraTS 2016 & 2017) portion of the dataset ([Task 1](https://drive.google.com/open?id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU)).
2. Untar the Task01_BrainTumour.tar to a folder.
3. Run the inference script:
```python evaluate_model.py --data_path $DECATHLON_ROOT_DIRECTORY --saved_model $SAVED_HDF5_FILE```, where `$DECATHLON_ROOT_DIRECTORY` is the root directory where you un-tarred the Decathlon dataset and $SAVE_HDF5_FILE is the name of the pre-trained Keras model.
e.g. 
```python evaluate_model.py --data_path ../../data/decathlon/Task01_BrainTumour/ --saved_model 3d_unet_decathlon_dice8621.hdf5```
replacing `--data_path` and `--saved_model` with your local paths/files.

There are many programs that will display [Nifti](https://nifti.nimh.nih.gov/) 3D files.  For the images below, we used the open-sourced package called [Mango](http://ric.uthscsa.edu/mango/).
The red overlay is the predictions from the model. The blue overlay is the ground truth masks. Any purple voxels are true positives.

![pred195](https://github.com/NervanaSystems/topologies/blob/master/3D_UNet/keras_training_only_version/images/BRATS_195_img.gif "BRATS image #195:  Purple voxels indicate a perfect prediction by the model. Red are false positives. Blue are false negatives")

 ![pred152](https://github.com/NervanaSystems/topologies/blob/master/3D_UNet/keras_training_only_version/images/BRATS_152.png "BRATS image #152:  Purple voxels indicate a perfect prediction by the model. Red are false positives. Blue are false negatives")

 ![pred426](https://github.com/NervanaSystems/topologies/blob/master/3D_UNet/keras_training_only_version/images/BRATS_426.png "BRATS image #426:  Purple voxels indicate a perfect prediction by the model. Red are false positives. Blue are false negatives")

REFERENCES:
1. Menze BH, Jakab A, Bauer S, Kalpathy-Cramer J, Farahani K, Kirby J, Burren Y, Porz N, Slotboom J, Wiest R, Lanczi L, Gerstner E, Weber MA, Arbel T, Avants BB, Ayache N, Buendia P, Collins DL, Cordier N, Corso JJ, Criminisi A, Das T, Delingette H, Demiralp Γ, Durst CR, Dojat M, Doyle S, Festa J, Forbes F, Geremia E, Glocker B, Golland P, Guo X, Hamamci A, Iftekharuddin KM, Jena R, John NM, Konukoglu E, Lashkari D, Mariz JA, Meier R, Pereira S, Precup D, Price SJ, Raviv TR, Reza SM, Ryan M, Sarikaya D, Schwartz L, Shin HC, Shotton J, Silva CA, Sousa N, Subbanna NK, Szekely G, Taylor TJ, Thomas OM, Tustison NJ, Unal G, Vasseur F, Wintermark M, Ye DH, Zhao L, Zhao B, Zikic D, Prastawa M, Reyes M, Van Leemput K. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

2. Bakas S, Akbari H, Sotiras A, Bilello M, Rozycki M, Kirby JS, Freymann JB, Farahani K, Davatzikos C. "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117
