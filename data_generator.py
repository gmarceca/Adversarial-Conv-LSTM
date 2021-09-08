import numpy as np
import keras
import pandas as pd
import abc
from window_functions import *
from label_smoothing import *
from helper_funcs import *
import os
import math
import sys
import random
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import glob
from plot_routines import *
from collections import defaultdict
from argparse import ArgumentParser

class IDsAndLabels(object):
    def __init__(self,):
        self.ids = defaultdict(dict)
        self.len = defaultdict(int)
    
    def __len__(self,):
        return self.len
    
    def generate_id_code(self, shot, index):
        return str(str(shot)+'/'+str(index))
    
    def add_id(self, machine, shot, k, transitions, elms, dithers):
        code = self.generate_id_code(shot, k)
        if code in self.ids[machine].keys():
            return
        else:
            self.ids[machine][code] = {'transitions': transitions, 'elms':elms, 'dithers': dithers}
            self.len[machine] += 1
    
    def get_sorted_ids(self, machine):
        return sorted(self.ids[machine].keys(), key = lambda ind: int(self.get_shot_and_id(ind)[1]))
    
    def get_ids(self, machine):
        return list(self.ids[machine].keys())
    
    def get_shot_and_id(self, ID):
        s_i = ID.split('/')
        return s_i[0], int(s_i[1])
            
    def get_label(self, ID, machine):
        return self.ids[machine][ID]
    
    def get_ids_and_labels(self, machine):
        pairs = []
        sorted_ids = sorted(self.get_ids(machine), key = lambda ind: int(self.get_shot_and_id(ind)[1]))
        for ind in sorted_ids:
            pairs += [[ind, self.get_label(ind, machine)]]
        return pairs

    def get_shots(self, machine):
        shots = []
        for ID in self.get_sorted_ids(machine):
            shot, ind = self.get_shot_and_id(ID)
            shots += [str(shot)]
        return set(shots)
    
class IDsAndLabelsLSTM(IDsAndLabels):
    def __init__(self,):
        IDsAndLabels.__init__(self)
    
    def add_id(self, machine, shot, k, elm_lab_in_dt, state_lab_in_dt): #state
        code = self.generate_id_code(shot, k)
        self.ids[machine][code] = (elm_lab_in_dt, state_lab_in_dt)
        self.len[machine] += 1
        
class LSTMDataGenerator():
    def __init__(self, shot_ids={}, batch_size=16, shuffle=True,
                 lstm_time_spread=150, epoch_size=400, conv_w_size=40, no_input_channels = 2,
                 gaussian_hinterval=10, no_classes=3, stride=10, labelers = ['marceca'], conv_w_offset=20, machine_id=['TCV'], normalization='minmax', data_augm=False):
       
        self.initialize(shot_ids, batch_size, shuffle,lstm_time_spread, epoch_size, conv_w_size, no_input_channels,gaussian_hinterval,
                        no_classes, stride, labelers,conv_w_offset,  machine_id, normalization)
        self.first_prepro_cycle()
        self.sec_prepro_cycle()
        self.third_prepro_cycle()
        self.data_augm = data_augm
     
        samples_per_type = batch_size//4
        
        # I leave this comment to remember self.ids definition
        #if no_classes == 2:
        #    self.ids = [self.ids_low, self.ids_high]
        #elif no_classes == 3:
        #    self.ids = [self.ids_low, self.ids_dither, self.ids_high]

        #self.class_generator = {}
        self.class_generator = defaultdict(dict)

        if self.no_classes==2:
            class_names = ['Low', 'High']
        elif self.no_classes==3:
            class_names = ['Low', 'Dither', 'High']
        for machine in self.machine_id:
            for ic, cl in enumerate(class_names):
                self.class_generator[machine][cl] = LSTMRandomDataFetcherEndtoEndWOffset(
                                                        self.ids[ic],
                                                        self.lstm_time_spread, 
                                                        samples_per_type,
                                                        self.shuffle,
                                                        self.conv_w_size,
                                                        self.no_input_channels,
                                                        self.shot_dfs[machine],
                                                        self.no_classes,
                                                        self.stride,
                                                        self.conv_w_offset,
                                                        machine)
        if self.no_classes==2:
            if len(self.machine_id) == 2: # Create a 50-50 balanced batch
                print('Getting generators for source and target machines! (LH)')
                self.sub_generators = [self.class_generator[self.machine_id[0]]['Low'], self.class_generator[self.machine_id[0]]['High'], self.class_generator[self.machine_id[1]]['Low'], self.class_generator[self.machine_id[1]]['High']]
            elif len(self.machine_id) == 1: # Nominal training
                print('Getting generators for source machine! (LH)')
                self.sub_generators = [self.class_generator[self.machine_id[0]]['Low'], self.class_generator[self.machine_id[0]]['High'], self.class_generator[self.machine_id[0]]['Low'], self.class_generator[self.machine_id[0]]['High']]
        
        elif self.no_classes==3:
            print('Getting generators for source machine! (LDH)')
            self.sub_generators = [self.class_generator[self.machine_id[0]]['Low'], self.class_generator[self.machine_id[0]]['Dither'], self.class_generator[self.machine_id[0]]['High'], self.class_generator[self.machine_id[0]]['Dither']]
    
    def initialize(self,shot_ids, batch_size, shuffle,lstm_time_spread, epoch_size, conv_w_size, no_input_channels, gaussian_hinterval,no_classes, stride, labelers,conv_w_offset,  machine_id, normalization):
        self.batch_size = batch_size
        # Get current directory path
        dirpath = os.getcwd()
        self.data_dir = defaultdict(str)
        self.normalization = normalization
        self.lstm_time_spread = int(lstm_time_spread)
        self.ids_low = IDsAndLabelsLSTM()
        self.ids_dither = IDsAndLabelsLSTM()
        self.ids_high = IDsAndLabelsLSTM()

        if no_classes == 2:
            self.ids = [self.ids_low, self.ids_high]
        elif no_classes == 3:
            self.ids = [self.ids_low, self.ids_dither, self.ids_high]
        
        # dic of tuples to assign a shot number to different labelers: {'TCV': (32195-labit, 32195-marceca, ...), 'AUG': (35550-marceca), ...}
        self.shot_ids = defaultdict(tuple)
        self.no_labelers = len(labelers)
        self.labelers = labelers
        self.machine_id = machine_id # In genreal this is a list of machines, ['TCV', 'AUG', etc]. The first in the list is the 'source' domain, the rest are the 'target' ones.
        for machine in self.machine_id:
            for s in shot_ids[machine]: # shot_ids = {'TCV': [32195, 64770, etc], 'AUG': [35550, etc]}
                for l in self.labelers:
                    self.shot_ids[machine] += (str(s) + '-' + str(l),)
            
            if 'algorithms' in dirpath: # Then it means I'm calling the function from algorithms/ConvLSTM/
                self.data_dir[machine] = os.path.join(dirpath, 'labeled_data/' + machine + '/')
            else: # I assume I'm calling the function from event-detection
                self.data_dir[machine] = os.path.join(dirpath, 'algorithms/ConvLSTM/labeled_data/' + machine + '/')
        self.length = epoch_size # This is the number of batches in one epoch
        
        # All dataframes will be stored here as dic of dics: {'TCV': {32195:df1, 64470:df2, ...}, 'AUG': {35550:df1}, ...}
        self.shot_dfs = defaultdict(dict)
        
        self.conv_w_size = conv_w_size
        self.no_input_channels = no_input_channels
        self.no_classes = no_classes
        self.stride = stride
        self.shuffle = shuffle
        self.conv_w_offset = conv_w_offset
        self.gaussian_hinterval = gaussian_hinterval
        
    def first_prepro_cycle(self,):
        gaussian_hinterval = self.gaussian_hinterval
        for machine in self.machine_id:
            for shot in self.shot_ids[machine]:
                fshot, fshot_times = load_fshot_from_labeler(shot, machine, self.data_dir[machine])
                fshot['sm_elm_label'], fshot['sm_non_elm_label'] = smoothen_elm_values(fshot.ELM_label.values, smooth_window_hsize=gaussian_hinterval)
                # Delete rows where LHD_label < 1 (might happen due to some issues during GUI labelling)
                fshot = fshot.drop(fshot[fshot.LHD_label < 1].index)
                # If no_classes == 2 (L or H) remove unexpected D labels
                if self.no_classes == 2:
                    fshot = fshot.drop(fshot[fshot.LHD_label == 2].index)
                # Transforms discrete labelling in continuous smoothed values in a gaussian_hinterval window
                fshot['sm_none_label'], fshot['sm_low_label'], fshot['sm_high_label'], fshot['sm_dither_label'] = smoothen_states_values_gauss(fshot.LHD_label.values,fshot.time.values, smooth_window_hsize=gaussian_hinterval)
                
                # CUTOFF to put all ELM labels at 0 where state is not H mode
                fshot.loc[fshot['LHD_label'] != 3, 'ELM_label'] = 0

                self.shot_dfs[machine][str(shot)] = fshot.copy()
    
    def sec_prepro_cycle(self,):
        for machine in self.machine_id:
            shots_id_list = self.shot_dfs[machine].keys()
            for shot in shots_id_list:
                shot_no = shot[:5]
                fshot = self.shot_dfs[machine][str(shot)].copy()
                fshot = normalize_signals_mean(fshot, machine, func=self.normalization)
                self.shot_dfs[machine][str(shot)] = fshot
        
    def third_prepro_cycle(self,):
        for machine in self.machine_id:
            for shot in self.shot_ids[machine]:
                fshot = self.shot_dfs[machine][str(shot)]
                for k in range(len(fshot)):
                    dt = fshot.iloc[k]
                    elm_lab_in_dt = get_elm_label_in_dt(dt)
                    state_lab_in_dt = get_state_labels_in_dt(dt)
                    if state_lab_in_dt[1] > .99: # If it's L mode
                        self.ids_low.add_id(machine, shot, k, elm_lab_in_dt, state_lab_in_dt)  
                    elif state_lab_in_dt[2] >.99: # If it's D mode
                        self.ids_dither.add_id(machine, shot, k, elm_lab_in_dt, state_lab_in_dt)
                    elif state_lab_in_dt[3] >.99: # If it's H mode
                        self.ids_high.add_id(machine, shot, k, elm_lab_in_dt, state_lab_in_dt)
                    #elif elm_lab_in_dt[0] != 0: # There are not validated ELMs in JET and so this (H states) is zero.
                    #    self.ids_high.add_id(machine, shot, k, elm_lab_in_dt, state_lab_in_dt)
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.length
    
    def get_batches(self, generators, index):
        batch_X_scalars, batch_y_states, batch_y_elms, batch_shots_ids, batch_ts, batch_X_start = None, None, None, None, None, None

        for sub_generator in generators: # Source generators
            X_scalars, y_states, y_elms, shots_ids, ts, X_start, machine_gen = sub_generator[[index]]

            # Fill batch
            batch_X_scalars = np.append(batch_X_scalars,X_scalars, axis = 0) if batch_X_scalars is not None else X_scalars
            batch_y_states = np.append(batch_y_states,y_states, axis = 0) if batch_y_states is not None else y_states
            batch_y_elms = np.append(batch_y_elms,y_elms, axis = 0) if batch_y_elms is not None else y_elms
            batch_shots_ids = np.append(batch_shots_ids,shots_ids, axis = 0) if batch_shots_ids is not None else shots_ids
            batch_ts = np.append(batch_ts,ts, axis = 0) if batch_ts is not None else ts
            batch_X_start = np.append(batch_X_start,X_start, axis = 0) if batch_X_start is not None else X_start

        aux = list(zip(
                        batch_X_scalars,
                        batch_y_states,
                        batch_y_elms,
                        batch_shots_ids,
                        batch_ts,
                        batch_X_start))
        if self.shuffle:
            random.shuffle(aux)
        batch_X_scalars, batch_y_states, batch_y_elms, batch_shots_ids, batch_ts, batch_X_start = zip(*aux)
        return np.asarray(batch_X_scalars), np.asarray(batch_y_states), batch_y_elms, batch_shots_ids, batch_ts, batch_X_start

    def __getitem__(self, index):

        batch_X_scalars_s, batch_y_states_s, batch_y_elms_s, batch_shots_ids_s, batch_ts_s, batch_X_start_s  = self.get_batches(self.sub_generators[0:2], index)
        batch_X_scalars_d, batch_y_states_d, batch_y_elms_d, batch_shots_ids_d, batch_ts_d, batch_X_start_d  = self.get_batches(self.sub_generators[2:4], index)

        ## Concatenate batches, 1st half source and 2nd half target
        ## Combine the labeled and unlabeled data along with the discriminative results
        #print('batch_X_scalars_s shape: ', batch_X_scalars_s.shape)
        combined_batchX = np.concatenate((batch_X_scalars_s, batch_X_scalars_s, batch_X_scalars_d, batch_X_scalars_d))
        #print('combined_batchX shape: ', combined_batchX.shape)
        batch_y_states_d = np.zeros(batch_y_states_s.shape)
        batch2Ys = np.concatenate((batch_y_states_s, batch_y_states_s, batch_y_states_d, batch_y_states_d))
        #print('batch2Ys shape: ', batch2Ys.shape)
        batchYd = np.concatenate((np.tile([0, 1], [combined_batchX.shape[0] // 2, combined_batchX.shape[1], 1]),np.tile([1, 0], [combined_batchX.shape[0] // 2, combined_batchX.shape[1], 1])))
        #print('batchYd shape: ', batchYd.shape)
        #return ({'input_signals':np.asarray(batch_X_scalars),'in_seq_start': np.asarray(batch_X_start),'times': np.asarray(batch_ts),'shots_ids': np.asarray(batch_shots_ids)},{'out_states':np.asarray(batch_y_states)})
        #return ({'input_signals':combined_batchX},{'out_states':batch2Ys})
        return ({'input_signals':combined_batchX},{'out_states':batch2Ys, 'domain_output':batchYd})
    
    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item
            if self.shuffle == True:
                print('Generator epoch finished, reshuffling...')
                self.on_epoch_end()
            
    def on_epoch_end(self):
        print('\n Generator epoch finished.')
        for generator in self.sub_generators:
            generator.on_epoch_end()
            
    
class LSTMRandomDataFetcherEndtoEndWOffset():
    def __init__(self, IDs_and_labels, lstm_time_spread=150, n_samples=16, shuffle = True, conv_w_size=41, no_input_channels = 3, shot_dfs=[], no_classes=3, stride=1, conv_w_offset=20, machine_id = 'TCV'):
        self.machine_id = machine_id # Only one machine here, TCV, AUG, etc
        self.no_input_channels = int(no_input_channels)
        self.n_samples = int(n_samples)
        self.IDs_and_labels = IDs_and_labels
        self.shuffle = shuffle 
        self.list_IDs = IDs_and_labels.get_ids(self.machine_id)
        self.indexes = np.arange(self.IDs_and_labels.len[self.machine_id])
        if self.shuffle:
            np.random.shuffle(self.indexes)   
        self.lstm_time_spread = lstm_time_spread
        self.windowed_scalars={}
        self.conv_w_size = int(conv_w_size)
        self.shot_dfs = shot_dfs
        self.no_classes = no_classes
        # Add input signals from dataframe into array
        self.readjust_windowed_indexes()
        self.stride = stride
        self.conv_w_offset = conv_w_offset
        
    def readjust_windowed_indexes(self,):
        for s in self.IDs_and_labels.get_shots(self.machine_id):
            shot = self.shot_dfs[s]
            self.windowed_scalars[s] = np.empty((len(shot)-self.conv_w_size, self.conv_w_size, self.no_input_channels))
            for k in range(self.conv_w_size):
                disloc = shot.iloc[k : len(shot) - self.conv_w_size + k]
                if self.no_input_channels == 1:
                    self.windowed_scalars[s][:, k, 0] = disloc.PD.values
                elif self.no_input_channels == 2:
                    self.windowed_scalars[s][:, k, 0] = disloc.FIR.values
                    self.windowed_scalars[s][:, k, 1] = disloc.PD.values
                elif self.no_input_channels == 3:
                    self.windowed_scalars[s][:, k, 0] = disloc.GWfr.values
                    self.windowed_scalars[s][:, k, 1] = disloc.PD.values
                    self.windowed_scalars[s][:, k, 2] = disloc.WP.values
    
    def data_generation(self, list_IDs_temp):
        spread = self.lstm_time_spread
        X_scalars_windowed = np.empty((self.n_samples, spread//self.stride, self.conv_w_size, self.no_input_channels))
        X_scalars = np.empty((self.n_samples, spread//self.stride, 1))
        X_start = np.empty((self.n_samples, 3))
        y_states = np.empty((self.n_samples, spread//self.stride, 3), dtype=float)
        y_elms = np.empty((self.n_samples, spread//self.stride, 2), dtype=float)
        ts = np.empty((self.n_samples, spread//self.stride, 1), dtype=float)
        shot_and_id = []
        for i, ID in enumerate(list_IDs_temp):
            shot, index = self.IDs_and_labels.get_shot_and_id(ID)
            offset = 0
            s_ind = index*1 + offset
            e_ind = index*1 + offset + self.lstm_time_spread
           
            min_id = self.conv_w_size + self.conv_w_offset
            # Ensure that start index is not below the minimum allowed
            if s_ind < min_id:
                temp = min_id - s_ind
                s_ind += temp
                e_ind += temp
            max_id = len(self.shot_dfs[shot]) - self.conv_w_size - self.conv_w_offset -1 
            # Ensure that end index is not above the maximum allowed
            if e_ind > max_id: 
                temp = e_ind - max_id
                e_ind -= temp
                s_ind -= temp
            assert s_ind >= min_id
            assert e_ind <= max_id
            
            conv_indexes = np.arange(s_ind-self.conv_w_size + self.conv_w_offset, e_ind-self.conv_w_size + self.conv_w_offset, self.stride)
            lstm_indexes = np.arange(s_ind, e_ind, self.stride) 
            scalars_windowed, scalars, states, elms, times = self.fetch_data_endtoend(shot, conv_indexes, lstm_indexes)
            
            if np.any(np.isnan(scalars_windowed)):
                print('nan, this should not happen', shot, index, s_ind, e_ind)
            if np.any(np.isnan(states)):
                print('nan in states, this should not happen', shot, index, s_ind, e_ind)
            
            X_scalars_windowed[i,] = np.asarray([scalars_windowed])
            y_states[i,] = states
            y_elms[i,] = elms
            shot_and_id.append([str(shot), str(s_ind), str(e_ind)])
            ts[i,] = times
            X_start[i,] = states[0]
        return X_scalars_windowed, y_states, y_elms, shot_and_id, ts, X_start, self.machine_id

    def fetch_data_endtoend(self, shot, conv_indexes, lstm_indexes):
        fshot = self.shot_dfs[shot]
        lstm_time_window = get_dt_and_time_window_windexes(lstm_indexes, fshot)
        try:
            states = get_states_in_window(lstm_time_window).swapaxes(0, 1)
        except:
            print('error in', shot, lstm_indexes)
            print(lstm_time_window.sm_low_label.values)
            checksum = lstm_time_window[['sm_low_label', 'sm_dither_label', 'sm_high_label']].sum(axis=1)
            print(np.where(checksum != 1))
            print(lstm_time_window.ix[np.where(checksum != 1)[0]])
            exit(0)
        elms = get_elms_in_window(lstm_time_window).swapaxes(0, 1)
        scalars_windowed = self.windowed_scalars[shot][conv_indexes, :, :]
        scalars = get_IP_in_window(lstm_time_window).swapaxes(0, 1)
        times = np.expand_dims(get_times_in_window(lstm_time_window), axis=0).swapaxes(0, 1)

        return scalars_windowed, scalars, states, elms, times
    
    def __getitem__(self, obj):
        index = obj[0]
        # Get n_samples  of a given conf state to start filling a batch

        # E.g n_samples = 16 sequences, each one starts in a given confinement state
        # index = 0, 1, 2, ... (call by the main data generator)
        # self.indexes = total indexes for a given conf state, (>> index*n_samples)
        # indexes will be [0, 1, ... 15], [16 ... 31], [31 ... 47], etc (if shuffle is set to false)
        assert(len(self.indexes))
        indexes = self.indexes[int((index*self.n_samples)%len(self.indexes)):int(((index+1)*self.n_samples)%len(self.indexes))]
        if len(indexes) == 0:
            # this may happen only when few data samples for a given conf state is available
            print('Not enough indexes to loop for the requested n_samples and batch_size')
            ids1 = self.indexes[(index*self.n_samples)%len(self.indexes):]
            ids2 = self.indexes[:(((index+1)*self.n_samples)%len(self.indexes))]
            indexes = np.concatenate([ids1, ids2])
        
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.data_generation(list_IDs_temp)
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.IDs_and_labels.len[self.machine_id])
        np.random.shuffle(self.indexes)


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Data generator for Conv-LSTM model')
    parser.add_argument("--baseline_test", action="store_true",
                        help="Perform unit test of baseline model")
    parser.add_argument("--adv_test", action="store_true",
                        help="Perform unit test of adversarial model")
    parser.add_argument("--plot", action="store_true",
                        help="Plot inputs")
    return parser


# For debugging random samples generator:
def main(args=None):
    parser = get_argparser()
    args = parser.parse_args(args)

    baseline_test = args.baseline_test # test baseline inputs
    adversarial_test = args.adv_test # test adversarial inputs
    plotter = args.plot # Plot inputs
    
    if baseline_test:
        machine_id = ['TCV']
        adversarial_training = False
        num_classes = 3
        labelers=['marceca']
        train_shots = {'TCV': [69129]}
    elif adversarial_test:
        machine_id = ['TCV', 'AUG']
        adversarial_training = True
        num_classes = 2
        labelers=['detrend']
        train_shots = {'TCV': [64770], 'AUG':[35604]}
    else:
        machine_id = ['TCV']

    if baseline_test and adversarial_test:
        raise ValueError('baseline and adversarial test cannot be both true')
    
    stride = 10
    conv_w_size =40
    lstm_time_spread = 2000
    conv_w_offset = 10
    gaussian_hinterval=5
    normalization = 'minmax'
    shuffle=False
    params_lstm_random = {
            'batch_size': 16*4,
            'lstm_time_spread': lstm_time_spread,
            'epoch_size': 64,
            'no_input_channels' : 2,
            'conv_w_size':conv_w_size,
            'gaussian_hinterval': gaussian_hinterval,
            'stride':stride,
            'labelers':labelers,
            'shuffle':shuffle,
            'conv_w_offset':conv_w_offset,
            'machine_id':machine_id,
            'normalization':normalization,
            'adversarial_training':adversarial_training,
            'no_classes': num_classes}
    
    training_generator = LSTMDataGenerator(shot_ids=train_shots, **params_lstm_random)         
    gen = next(iter(training_generator))
    
    output_states = []
    input_signals = []
    times = []
    input_weights_clf = []
    input_weights_dom = []
    
    counter = 0
    for batch in gen:
        inputs = batch[0]
        targets = batch[1]
        weights = batch[2]
        print('batch X: ', np.asarray(inputs['input_signals']).shape)
        for sample in range(params_lstm_random['batch_size']):
            counter += 1
            output_states += [np.asarray(targets['out_states'][sample])]
            input_signals += [np.asarray(inputs['input_signals'][sample])]
            times += [np.asarray(inputs['times'][sample])]
            if type(weights) == list:
                input_weights_clf += [np.asarray(weights[0][sample])]
                input_weights_dom += [np.asarray(weights[1][sample])]
            else:
                input_weights_clf += [np.asarray(weights[sample])]
        if counter == 4*params_lstm_random['batch_size']:
            break
        if np.any(inputs['input_signals'] == float('nan')):
            break
    output_states = np.asarray(output_states)
    input_signals = np.asarray(input_signals)
    times = np.asarray(times)
    input_weights_clf = np.asarray(input_weights_clf)
    input_weights_dom = np.asarray(input_weights_dom)
    # Get current directory path
    dirpath = os.getcwd()
    import scipy.io
    if baseline_test:
        if 'algorithms' in dirpath: # Then it means I'm calling the function from algorithms/ConvLSTM/
            scipy.io.savemat(os.path.join(dirpath, 'experiments/baseline_16042021_exp9/input_signal_TCV_69129_test.mat'), {'input_signals': input_signals[0, :, 0, :]})
        else: # I assume I'm calling the function from event-detection
            scipy.io.savemat(os.path.join(dirpath, 'algorithms/ConvLSTM/experiments/baseline_16042021_exp9/input_signal_TCV_69129_test.mat'), {'input_signals': input_signals[0, :, 0, :]})

    elif adversarial_test:
        if 'algorithms' in dirpath: # Then it means I'm calling the function from algorithms/ConvLSTM/
            scipy.io.savemat(os.path.join(dirpath, 'experiments/adversarial_test/sample_weights_test.mat'), {'input_weights_clf': input_weights_clf[0:64,0], 'input_weights_dom': input_weights_dom[0:64,0]})
        else: # I assume I'm calling the function from event-detection
            scipy.io.savemat(os.path.join(dirpath, 'experiments/adversarial_test/sample_weights_test.mat'), {'input_weights_clf': input_weights_clf[0:64,0], 'input_weights_dom': input_weights_dom[0:64,0]})
if __name__ == '__main__':
    main()
