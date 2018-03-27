# Usage: ./single_brain_samples.sh
# Generates predicted masks on the test set with dim 128x128 and
# raw brain images at full resolution.


data_dir='/home/bduser/data_test/MICCAI_BraTS17_Data_Training/'
section='HGG/'
sample_dir='/home/bduser/data_test/'
train_test_split=0.80 # percent of the dataset to use for training
declare -i fin_msk_size=128 # must be 128 to work with current U-Net
declare -i fin_img_size=128 # can be larger than 128 if desired, is re-made after inference

source activate tf

# Add scans in list below to generate single brain samples
for sample in 'Brats17_TCIA_296_1' 'Brats17_TCIA_607_1'; do

        cp -r $data_dir$section$sample $sample_dir

        mkdir $sample_dir$sample/data

        time python sample_converter.py $data_dir $sample $train_test_split $fin_msk_size $sample_dir$sample/data/

        time python sample_inference.py $sample_dir$sample/data/

        rm $sample_dir$sample/data/*test.npy

        rm $sample_dir$sample/data/*train.npy

        time python sample_converter.py $data_dir $sample $train_test_split $fin_img_size $sample_dir$sample/data/

done

source deactivate
