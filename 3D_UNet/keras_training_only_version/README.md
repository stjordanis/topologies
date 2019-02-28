# 3D U-Net for Medical Decathlon Dataset

Trains a 3D U-Net on the brain tumor segmentation (BraTS) subset of the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) dataset.

Steps to train a new model:

1. Go to the [Medical Segmentation Decathlon](http://medicaldecathlon.com) website and download the [BraTS subset](https://drive.google.com/file/d/1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU/view?usp=sharing). The dataset has the Creative Commons Attribution-ShareAlike 4.0 International [license](https://creativecommons.org/licenses/by-sa/4.0/).
2. Untar the "Task01_BrainTumour.tar" file (e.g. `tar -xvf Task01_BrainTumour.tar`)
3. Create a Conda environment with TensorFlow. Command: `conda create -c anaconda -n decathlon pip python=3.6 tensorflow keras tqdm h5py psutil`
4. Enable the new environment. Command: `conda activate decathlon`
5. Install the package [nibabel](http://nipy.org/nibabel/). Command: `pip install nibabel`
6. Run the command `python train.py --data_path $DECATHLON_ROOT_DIRECTORY`, where `$DECATHLON_ROOT_DIRECTORY` is the root directory where you un-tarred the Decathlon dataset.


Steps to evaluate a pre-trained 3D U-Net model.

1. Download the [Medical Decathlon dataset](http://medicaldecathlon.com/). Specifically, this model was trained on the brain tumor segmentation (BraTS 2016 & 2017) portion of the dataset ([Task 1](https://drive.google.com/open?id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU)).
2. Untar the Task01_BrainTumour.tar to a folder.
3. Run the inference script:
`python evaluate_model.py --data_path $DECATHLON_ROOT_DIRECTORY --saved_model $SAVED_HDF5_FILE`, where `$DECATHLON_ROOT_DIRECTORY` is the root directory where you un-tarred the Decathlon dataset and $SAVE_HDF5_FILE is the name of the pre-trained Keras model.
e.g. 
`python evaluate_model.py --data_path ../../../data/decathlon/Task01_BrainTumour/ --saved_model 3d_unet_decathlon_dice8602.hdf5`
replacing `--data_path` and `--saved_model` with your local paths/files.

 ![predC](https://github.com/NervanaSystems/topologies/blob/master/3D_UNet/keras_training_only_version/images/predC.png)

 ![predB](https://github.com/NervanaSystems/topologies/blob/master/3D_UNet/keras_training_only_version/images/predB.png)
