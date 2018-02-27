age: ./single_brain_samples.sh
# Generates predicted masks on the test set with dim 128x128 and
# raw brain images at full resolution.


data_dir='/home/bduser/data_test/MICCAI_BraTS17_Data_Training/'
section='HGG/'
sample_dir='/home/bduser/data_test/'

source activate tf

# Add scans in list below to generate single brain samples
for sample in 'Brats17_TCIA_277_1'; do

        cp -r $data_dir$section$sample $sample_dir

        mkdir $sample_dir$sample/data

        time python sample_converter.py $data_dir $sample 128 $sample_dir$sample/data/

        time python sample_sanity_check.py $sample_dir$sample/data/

        rm $sample_dir$sample/data/*test.npy

        rm $sample_dir$sample/data/*train.npy

        time python sample_converter.py $data_dir $sample 240 $sample_dir$sample/data/

        rm $sample_dir$sample/data/*train.npy

done

source deactivate
