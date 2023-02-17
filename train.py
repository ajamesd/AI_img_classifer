import argparse
import torch
import data_loader
import model_functions


def main():
    
    parser = argparse.ArgumentParser("Trains a VGG((13/16/19)) to indentify and predict types of flowers")
    
    parser.add_argument('data_directory', help='file path to asset data',
                        metavar='DIR', default ='flower_data')
    
    parser.add_argument('--save_dir',help='Checkpoint save directory', metavar='DIR',
                        default='./', dest='save_dir')

    parser.add_argument('--arch', action='store', help='Version of VGG neural network architecture - vgg13, vgg16 or vgg19',
                        default='vgg16', dest='arch')

    parser.add_argument('--hidden_units', action='store', type=int,
                        help='Hidden units hyperparameter. Some recommended options are 512, 1024 or 2048',
                        default=512, dest='hidden_units')

    parser.add_argument('--learning_rate', action='store', type=float,
                        help='Learning rate Hyperparameter - 0.001 to 0.003 is the recommended range', default=.003,
                        dest='learning_rate')

    parser.add_argument('--epochs', action='store', type=int,
                        help='Number of epochs hyperparameter', dest='epochs',
                        default=2)

    parser.add_argument('--gpu', action='store_true', help='Option to run model training on gpu - recommended',
                        dest='gpu', default=False)    
    
    
    args = parser.parse_args()
    

    
    
    
    if args.gpu == True:
        gpu = True
    else:
        gpu = False
        
    data_set, class_to_idx  = data_loader.load_data(args.data_directory)
    
    model_functions.model_train(data_set, class_to_idx, args.hidden_units, args.learning_rate, args.epochs, args.arch, gpu, args.save_dir)
    
    
if __name__ == "__main__":
        main()