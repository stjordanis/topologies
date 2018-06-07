# Assuming you have created a virtual (conda) environment with TensorFlow

train_steps=1   # Number of steps to train model
batch_size=64   # Batch size for inference

# There are several NMT models to choose:
# --model_type {ListenAttendSpell,NMTBig,NMTMedium,NMTSmall,SeqTagger,Transformer,TransformerAAN,TransformerBig}
nmt_model=NMTSmall 

rm -rf OpenNMT-tf

clear

echo "$(tput setaf 2)Cloning OpenNMT from GitHub$(tput setaf 7)"
echo " "
# Clone the latest version of openNMT
git clone https://github.com/OpenNMT/OpenNMT-tf.git
cd OpenNMT-tf
echo "Installing OpenNMT into python environment"
pip install pyonmttok
python setup.py install

clear

echo "$(tput setaf 2)Building vocabulary for German/English translation$(tput setaf 7)"
# Now build the German-English vocabularies from the standard dataset
onmt-build-vocab --size 50000 --save_vocab data/toy-ende/src-vocab.txt data/toy-ende/src-train.txt
onmt-build-vocab --size 50000 --save_vocab data/toy-ende/tgt-vocab.txt data/toy-ende/tgt-train.txt

clear

echo "$(tput setaf 2)Training model $nmt_model for $train_steps step(s) just to get random weights for model$(tput setaf 7)"
echo " "
echo " "

# Change training to a single step just to do inference on the model
sed -ri "s/^(\s*)(train_steps\s*:\s*1000000\s*$)/\1train_steps: $train_steps/" config/opennmt-defaults.yml

# There are several types of NMT models to test:

# Train the model for 1 step
onmt-main train_and_eval --model_type $nmt_model --config config/opennmt-defaults.yml config/data/toy-ende.yml

clear

echo "$(tput setaf 2)Creating predictions for model $nmt_modelwith batch size of $batch_size$(tput setaf 7)"
echo " "
echo " "
sed -ri "s/^(\s*)(batch_size\s*:\s*30\s*$)/\1batch_size: $batch_size/" config/opennmt-defaults.yml

# Perform inference on the standard testing German/English dataset
onmt-main infer --log_prediction_time --config config/opennmt-defaults.yml config/data/toy-ende.yml --features_file data/toy-ende/src-test.txt

echo " "
echo "$(tput setaf 4)Finished inference script$(tput setaf 7)"

