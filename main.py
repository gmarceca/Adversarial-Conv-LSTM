# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, warnings
gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']=gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
warnings.filterwarnings('ignore')

import numpy as np
import argparse
import gc
import matplotlib
matplotlib.use('Agg')
import util
import utilDANN
import utilNNModel
import utilDANNModel
import utilGetData
from keras import backend as K

util.init()
K.set_image_data_format('channels_last')

if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    #config = tf.compat.v1.ConfigProto()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #sess = tf.compat.v1.Session(config=config)
    sess = tf.Session(config=config)
    K.set_session(sess)

# -----------------------------------------------------------------------------
def train_dann(input_shape, num_labels, config, experiment):
    
    print('Getting data...')
    X_scalars_test, GT, fshots, shots = utilGetData.get_data()
    print('Done')
    machines = ['TCV', 'JET'] # source and target machines
    params_lstm_random = {
            'batch_size': config.batch_size,
            'lstm_time_spread': config.lstm_time_spread,
            'epoch_size': config.epoch_size,
            'no_input_channels' : config.no_input_channels,
            'conv_w_size': config.conv_w_size,
            'gaussian_hinterval': config.gaussian_hinterval,
            'stride': config.stride,
            'labelers': [config.labelers],
            'shuffle': config.shuffle,
            'conv_w_offset': config.conv_w_offset,
            'normalization': config.normalization,
            'no_classes': config.num_classes}

    if config.type=='adv':
        print('Getting adv model ...')
        config.batch_size = config.batch_size*2
        dann = utilDANNModel.DANNModel(config.model, input_shape, num_labels, config.batch_size)
    else:
        print('Getting std model ...')
        dann = utilNNModel.NNModel(config.model, input_shape, num_labels, config.batch_size)

    # I leave this comment since it will help in the future
    #dann_model = dann.build_tsne_model()
    #dann_vis = dann.build_dann_model()

    # Create dir with the id name of the experiment
    train_dir = './experiments/' + experiment
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    print('Will save this model in', train_dir)
    
    if config.load == False:
        utilDANN.train_dann(dann, None, None, config, experiment, params_lstm_random, machines, target_x_test=X_scalars_test, target_y_test=GT, test_shots=shots, fshots=fshots)
    else:
        print(' Load pre-trained model')
        dann.dann_model.load_weights(train_dir + '/model_checkpoints/' + 'weights.64.h5')  # Load the save weights for a given epoch
        print('Start fine-tuning')
        utilDANN.train_dann(dann, None, None, config, experiment, weights_filename, params_lstm_random, machines, target_x_test=X_scalars_test, target_y_test=GT, test_shots=shots, fshots=fshots)

    gc.collect()



# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DANN')

    group0 = parser.add_argument_group('Training type')
    group0.add_argument('-type',   default='adv', type=str,  help='Training type: adversarial (adv) or normal (std)')
    group0.add_argument('-model', default=99,         type=int,        help='Model number (1,2,3,4)')
    
    group1 = parser.add_argument_group('Input data parameters')
    group1.add_argument('-e',default=100,type=int,dest='epochs',help='Number of epochs')
    group1.add_argument('-stride',default=10,type=int, help='CNN-LSTM stride')
    group1.add_argument('-conv_w_size', default=40,type=int, help='CNN window size')
    group1.add_argument('-lstm_time_spread',default=2000, type=int, help='input seq length')
    group1.add_argument('-conv_w_offset',default=10,type=int, help='input seq offset')
    group1.add_argument('-gaussian_hinterval',default=5,type=int, help='smooth window size for labels')
    group1.add_argument('-normalization',default='z_GCS',type=str, help='smooth window size for labels')
    group1.add_argument('-shuffle',default=True,type=bool, help='shuffle data generator')
    group1.add_argument('-num_classes',default=2,type=int, help='number of confinement modes')
    group1.add_argument('-labelers',default='marceca',type=str, help='labeler name/type')
    group1.add_argument('-batch_size',default=64,type=int, help='batch size')
    group1.add_argument('-epoch_size',default=64,type=int, help='epoch size')
    group1.add_argument('-no_input_channels',default=3,type=int, help='number of input signals')

    parser.add_argument('-lda',      default=0.03,    type=float,    help='Reversal gradient lambda')
    parser.add_argument('-lda_inc',      default=0.001,    type=float,    help='Reversal gradient lambda increment')
    parser.add_argument('-lr',        default=1.0,     type=float,    help='Learning rate')
    
    parser.add_argument('-set',        default=1,     type=int,    help='Set of shots used for adv')

    parser.add_argument('--tsne',    action='store_true',               help='Activate TSNE')
    parser.add_argument('--aug',    action='store_true',               help='Use data augmentation')
    parser.add_argument('--load',  action='store_true',               help='Load weights.')
    parser.add_argument('--v',         action='store_true',      help='Activate verbose.')
    parser.add_argument('-gpu',    default='0',       type=str,         help='GPU')
    args = parser.parse_args()

    input_shape = (None, 40, 3)
    num_labels = 3
    
    if args.type == 'std':
        args.lda = 0.0
        args.lda_inc = 0.0

    experiment = 'l_{}_inc_{}_{}_set{}_wodet'.format(str(args.lda), str(args.lda_inc), str(args.type), str(args.set))

    train_dann(input_shape, num_labels, args, experiment)
