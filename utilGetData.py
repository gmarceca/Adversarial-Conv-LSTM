import keras
import numpy as np
import pickle
from helper_funcs import load_dic, remove_current_1MA, remove_current_30kA, remove_current_370kA, remove_no_state, remove_disruption_points, normalize_current_MA, normalize_signals_mean
import pandas as pd
from collections import defaultdict
from glob import glob
import resource
import gc
import json

def get_data():
    """
    Function that gets the data to evaluate the model
    """
    test_set_aug = [35248, 35274, 35275, 35529, 35530, 35531, 35532, 35536, 35538, 35540, 35556, 35557, 35564, 35582, 35584, 35604, 35607, 35837, 35852, 35967, 35972, 35975] #35537, 35539, 35561] removed for the moment because no DML
    test_set_jet = [97927, 97803, 96532, 97828, 97830, 97832, 97835, 97971, 97465, 97466, 97468, 97469, 97473, 94785, 97476, 97477, 96713, 97745, 98005, 96993, 96745, 94315, 97396, 96885, 97398, 97399, 94968, 94969, 94971, 94973]
    test_set_tcv = [61057, 64770, 57093, 64774, 57095, 61714, 64662, 62744, 59065, 64060, 59073, 60097, 61010, 61274, 58460, 61021, 64369, 61043]

    shots = defaultdict(list)
    labeler = 'detrend'
    normalization = 'z_GCS'
    conv_window_size = 40
    conv_w_offset = 10
    no_input_channels = 3
    X_scalars_test = defaultdict(list)
    GT = defaultdict(list)
    fshots = defaultdict(list)
    states_pred_concat = defaultdict(list)
    ground_truth_concat = defaultdict(list)
        
    shots['TCV'] = test_set_tcv
    shots['JET'] = test_set_jet
    shots['AUG'] = test_set_aug
    machines = ['TCV', 'JET', 'AUG']
    for machine in machines:
        for i, shot in zip(range(len(shots[machine])), shots[machine]):
            fshot = pd.read_csv('./labeled_data/' + machine + '/' + labeler + '/' + machine + '_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
            shot_df = fshot.copy()
            if machine == 'JET':
                shot_df = remove_current_1MA(shot_df)
            elif machine == 'TCV':
                shot_df = remove_current_30kA(shot_df)
            elif machine == 'AUG':
                shot_df = remove_current_370kA(shot_df)
            else:
                raise ValueError('Machine_id {} not stored in the database'.format(machine))
            shot_df = remove_no_state(shot_df)
            shot_df = remove_disruption_points(shot_df)
            shot_df = shot_df.reset_index(drop=True)
            shot_df = normalize_current_MA(shot_df)
            shot_df = normalize_signals_mean(shot_df, machine, func=normalization)
            
            stride=10
            length = int(np.ceil((len(shot_df)-conv_window_size)/stride))
            X_scalars_single = np.empty((length, conv_window_size, no_input_channels))
            intersect_times = np.round(shot_df.time.values,5)
            intersect_times = intersect_times[conv_window_size-conv_w_offset:len(intersect_times)-conv_w_offset]
            for j in np.arange(length):
                vals = shot_df.iloc[j*stride : conv_window_size + j*stride]
                scalars = np.asarray([vals.GWfr, vals.PD, vals.WP]).swapaxes(0, 1)
                assert scalars.shape == (conv_window_size, no_input_channels)
                X_scalars_single[j] = scalars
            X_scalars_test[machine] += [X_scalars_single]
            fshot_sliced = shot_df.loc[shot_df['time'].round(5).isin(intersect_times)]
            GT[machine] += [fshot_sliced['LHD_label'].values[0::stride]]
            fshots[machine] += [fshot_sliced]
            pass
        pass
    return X_scalars_test, GT, fshots, shots
