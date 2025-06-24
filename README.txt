Directories:

MLR-src: source code for the MLR model (mVAE, label networks, SVMs...)
checkpoints: saved weight sets for models including autoencoder, SVM and label networks
data:  Datasets  (mnist, emnist, quickdraw, cifar, bengali characters)
mnist_data:  Datasets (redundant with data, need to trim down)
simulation_src: source code for all specific  simulations
simulations: output of simulation runs
documentation: packages to install
training_samples:  diagnostic images during training are placed here


These are key files

Root directory:
 - Training.py:  Wrapper function for training a new model from scratch
 - plotting.py:  Wrapper function for running specific simulations


MLR-2.0 model is outlined in MLR_src/    
 - mVAE.py: defines the cropped encoder, cropped decoder, and retinal decoder NNs in the VAE_CNN class. 
        Training objective functions are defined below this class
 - BP_functions.py: defines the Binding Pool memory functions
 - classifiers.py: defines the SVM classifiers that operate on MLR-2.0's latent representations
 - label_network.py: defines the label network to project one-hot vectors into latent representations
 - dataset_builder.py: defines the modified dataset class for training
 - train_mVAE.py:  function to manage training of the mVAE

Simulation files are defined in simulation_src
 - 