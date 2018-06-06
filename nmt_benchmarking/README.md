# Benchmarking script for neural machine translation.

This uses the [OpenNMT TensorFlow](http://opennmt.net/) project to benchmark deep learning RNN topologies.

Just run the bash script in whatever environment you have TensorFlow installed.

It will download the OpenNMT repo, install it, train a model for 1 step, and then perform inference with a batch size of 64. The train and test set is the open German/English corpus.


