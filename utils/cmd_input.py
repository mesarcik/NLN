import argparse
from coolname import generate_slug as new_name

"""
    Pretty self explanatory, gets arguments for training and adds them to config
"""
parser = argparse.ArgumentParser(description='Train generative anomaly detection models')

parser.add_argument('-limit',metavar='-l', type=str, default='None',
                    help = 'Limit on the number of samples in training data ')
parser.add_argument('-anomaly_class',metavar='-a', type=str,  default=2,
                    help = 'The labels of the anomalous class')
parser.add_argument('-percentage_anomaly', metavar='-p', type=float, default=0,
                    help = 'The percentage of anomalies in the training set')
parser.add_argument('-epochs', metavar='-e', type=int, default=100,
                    help = 'The number of epochs for training')
parser.add_argument('-latent_dim', metavar='-ld', type=int, default=2,
                    help = 'The latent dimension size of the AE based models')
parser.add_argument('-neighbors', metavar='-n', type=int, nargs='+', default=[2,4,5,6,7,8,9],
                    help = 'The maximum number of neighbours for latent reconstruction')
parser.add_argument('-radius', metavar='-r', type=float, nargs='+', default=[0.1,0.5,1,2,5,10],
                    help = 'The radius of the unit circle for finding neighbours in frNN')
parser.add_argument('-algorithm', metavar='-nn', type=str, choices={"radius", "knn"}, 
                    default='radius', help = 'The algorithm for calculating neighbours')
parser.add_argument('-data', metavar='-d', type=str, default='MNIST',
                    help = 'The dataset for training and testing the model on')

args = parser.parse_args()
args.model_name = new_name()

if args.data == 'MNIST' or args.data == 'FASHION_MNIST':
    args.input_shape =(28,28,1)
elif args.data == 'CIFAR10'
    args.input_shape =(32,32,3)

if args.limit == 'None':
    args.limit = None
else: 
    args.limit =  int(args.limit)

if ((args.data == 'MNIST') or 
    (args.data == 'CIFAR10') or 
    (args.data == 'FASHION_MNIST')): 

    args.anomaly_class = int(args.anomaly_class)

