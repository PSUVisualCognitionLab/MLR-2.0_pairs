Requirements to install:

conda install

conda create --name yourname37 python=3.7
conda activate yourname37
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install scikit-learn
conda install matplotlib
conda install tqdm imageio ipython opencv pandas

MLR-2.0 model is outlined in MLR_src/:
 - mVAE.py: defines the cropped encoder, cropped decoder, and retinal decoder NNs in the VAE_CNN class. Training objective functions are defined belwo this class
 - BP_functions.py: defines the Binding Pool memory functions
 - classifiers.py: defines the SVM classifiers that operate on MLR-2.0's latent representations
 - label_network.py: defines the label network to project one-hot vectors into latent representations
 - dataset_builder.py: defines the modified dataset class for training

Run Training.py to train a new model instance.