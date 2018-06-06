# Assuming you have created a virtual (conda) environment with TensorFlow
rm -rf OpenNMT-tf

echo "Cloning OpenNMT from GitHub"
echo " "
# Clone the latest version of openNMT
git clone https://github.com/OpenNMT/OpenNMT-tf.git
cd OpenNMT-tf
echo "Installing OpenNMT into python environment"
python setup.py install

clear

echo "Building vocabulary for German/English translation"
# Now build the German-English vocabularies from the standard dataset
onmt-build-vocab --size 50000 --save_vocab data/toy-ende/src-vocab.txt data/toy-ende/src-train.txt
onmt-build-vocab --size 50000 --save_vocab data/toy-ende/tgt-vocab.txt data/toy-ende/tgt-train.txt

clear

echo "Training for a single step just to get random weights for model"
echo " "
echo " "
# Change training to a single step just to do inference on the model
sed -ri 's/^(\s*)(train_steps\s*:\s*1000000\s*$)/\1train_steps: 1/' config/opennmt-defaults.yml

# Train the model for 1 step
onmt-main train_and_eval --model_type NMTSmall --config config/opennmt-defaults.yml config/data/toy-ende.yml


clear

echo "Creating predictions with batch size of 64"
echo " "
echo " "
sed -ri 's/^(\s*)(batch_size\s*:\s*30\s*$)/\1batch_size: 64/' config/opennmt-defaults.yml

# Perform inference on the standard testing German/English dataset
onmt-main infer --log_prediction_time --config config/opennmt-defaults.yml config/data/toy-ende.yml --features_file data/toy-ende/src-test.txt

echo " "
echo "Finished inference script"

