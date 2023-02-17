AI Programming with Python Final Project: Developing an Image Classifier with Deep Learning
Please see the repo for the files required to train one of three Neural Network models: VGG 13, VGG 16 and VGG 19.

Note that training data is not included in this repo

TRAIN

Train.py should be run initally to train the selected model.

The required input is python train.py data_directory (In this case data_directory is flower_data).

Additional optional arguments for train.py are:

Set directory to save checkpoints: python train.py data_dir --save_dir save_director
Choose architecture: python train.py data_dir --arch “vgg13”
Set hyperparameters: python train.py data_dir --learning_rate 0.01 – hidden_units 512 – epochs 20
Use GPU for training: python train.py data_dir --gpu
PREDICT
This predicts the type of flower in a given image. The data returned represents the probability the image is one of 5 flowers.

Usage:

predict.py is the baseline command

Required: a single imagine and corresponding filepath (flower1.jpg) and the filepath and name of the checkpoint (.pth) created as the output from

Basic usage: python predict.py /path/to/image checkpoint

Options:

Return top K most likley classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
Written with StackEdit.
