# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from lstm_data_generator import *
import json
from custom_callbacks import ValidationKappaScore

# ----------------------------------------------------------------------------
def train_dann(dann_builder, target_x_train, target_y_train, config, experiment, params_lstm_random, machines, initial_hp_lambda=0.01,
                                 target_x_test = None, target_y_test = None, test_shots=None, fshots=None):
    
    initial_hp_lambda = config.lda
    #initial_hp_lambda = 0.0 # Use this setting for fine-tuning
    hp_lambda_increment = config.lda_inc
    type_ = config.type
    batch_size = config.batch_size
    nb_epochs = config.epochs
    
    params_random_source = {}
    train_shots = {}
    for im, machine in enumerate(machines):
        # Get train and val shots split from json db
        with open('train_and_val_shots.json', 'r') as a:
            shots_db = json.load(a)
        try:
            if machine == 'JET': #FIXME
                # Set 1
                #train_shots[machine] = [96993, 97832, 97468, 97398, 97835, 97830, 94785]
                train_shots[machine] = [96993]
            elif machine == 'AUG':
                # Set 2 # FIXME
                train_shots[machine] = [35248, 35274, 35532, 35557, 35564, 35837, 35972]
            else:
                train_shots[machine] = shots_db[machine][0]['train_shots']
        except:
            print("machine_id {} is not in json database".format(machine))
            raise
   
    # First machine is source (with labels) and second is target (without labels)
    params_random_source = {'shot_ids': train_shots, 'machine_id': machines}
    params_random_source.update(params_lstm_random)

    print('params source: ', params_random_source)
    
    src_generator = LSTMDataGenerator(**params_random_source)
    src_generator = next(iter(src_generator))
    
    if type_=='adv': 
        dann_builder.grl_layer.set_hp_lambda(initial_hp_lambda)
    
        print('Starting adv training ...')
        # Update learning rates
        lr = float(K.get_value(dann_builder.opt.lr))* (1. / (1. + float(K.get_value(dann_builder.opt.decay)) * float(K.get_value(dann_builder.opt.iterations)) ))
        print(' - Lr:', lr, ' / Lambda:', dann_builder.grl_layer.get_hp_lambda())
        dann_builder.grl_layer.increment_hp_lambda_by(hp_lambda_increment)
  
    checkpoint_dir = './experiments/' + experiment +'/model_checkpoints/'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saveCheckpoint = keras.callbacks.ModelCheckpoint(filepath= checkpoint_dir + 'weights.{epoch:02d}.h5', period=2, save_weights_only=True)

    # Evaluate kappa score at epoch end
    kappascore = ValidationKappaScore(experiment, machines, 'z_GCS', False, config)

    dann_builder.dann_model.fit_generator(generator = src_generator, steps_per_epoch=64, epochs=10, callbacks=[kappascore], shuffle=False, verbose=0)

