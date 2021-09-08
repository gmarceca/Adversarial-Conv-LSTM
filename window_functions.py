import numpy as np
import pandas as pd
import tensorflow as tf

# def get_vals_w_index(k, w_spread, fshot):
#     current_dt = fshot.iloc[k]
#     window = fshot.iloc[k-w_spread:k+w_spread+1]
#     return (current_dt, window)
trans_ids = ['LH', 'HL', 'DH', 'HD', 'LD', 'DL', 'no_trans']
def get_dt_and_time_window(k, lstm_time_spread, fshot):
    # print 'k', k
    current_dt = fshot.iloc[k]
    window = fshot.iloc[k-lstm_time_spread:k+lstm_time_spread+1]
    return (current_dt, window)

def get_dt_and_time_window_wstartindex(start_index, end_index, fshot):
    window = fshot.iloc[start_index:end_index]
    return window

def get_dt_and_time_window_windexes(indexes, fshot):
    window = fshot.iloc[indexes]
    return window

def get_states_in_window(time_window):
    # return np.asarray([time_window.state]) #includes dither!
    # print 'time_window'
    # print time_window.low, time_window.high, time_window.dither
    # print np.asarray([time_window.low.values, time_window.high.values, time_window.dither.values])
    l, d, h = time_window.sm_low_label.values, time_window.sm_dither_label.values, time_window.sm_high_label.values
    for k in range(len(l)):
        low, dither, high = l[k], d[k], h[k]
        assert(round(low+dither+high, 3)==1)
    return np.asarray([l, d, h]) #this gets the dither target for the RNN time_window.sm_none_label.values,


def get_times_in_window(time_window):
    return np.asarray(time_window.time.values)

def get_elms_in_window(time_window):
    return np.asarray([time_window.sm_elm_label.values, time_window.sm_non_elm_label.values])

def get_state_in_dt(dt):
    # return np.asarray(dt.state)
    return np.asarray([dt.low, dt.high, dt.dither])

def get_dt_and_window(k, w_spread, fshot, conv_offset):
    # print 'k', k
    current_dt = fshot.iloc[k]
    window = fshot.iloc[k-w_spread + conv_offset:k + conv_offset]
    return (current_dt, window)
    
def get_raw_signals_in_window(window):  
    # return (np.asarray(window.FIR.values), np.asarray(window.DML.values), np.asarray(window.PD.values))
    return np.asarray([window.FIR.values, window.DML.values, window.PD.values, window.IP.values])
    # assert len(window.fir) == 101
    # return (weighted_avg_and_normalize(window.fir), weighted_avg_and_normalize(window.DML), weighted_avg_and_normalize(window.PD))

def get_IP_in_window(window):
    return np.asarray([window.IP.values])

def get_raw_signals_in_lstm_window(lstm_window):  
    return np.asarray([lstm_window.fir.values, lstm_window.PD.values])#, lstm_window.Ip_hf.values]) #np.asarray(lstm_window.DML.values), 

def weighted_avg_and_normalize(vals):
    # print vals.shape
    # assert len(vals == 50)
    a1 = np.linspace(0, 1, 51)
    a2 = np.linspace(1, 0, 51)
    avg_weights = np.concatenate((a1, a2[1:]))
    avg_weights = avg_weights[0:vals.shape[0]]
    # print vals
    # print avg_weights
    # print avg_weights.shape
    avgd_val = np.average(vals, weights = avg_weights)
    # print avgd_val
    subt = vals - avgd_val
    inf = subt - np.min(subt)
    stand = inf / np.max(inf)
    norm = stand * 2 - 1
    return norm

def get_signals_in_dt(current_dt):
    return (current_dt.Ip_hf, current_dt.tot_p)

def get_scalar_signals_in_window(lstm_time_window):
    return np.asarray(lstm_time_window.Ip_hf, lstm_time_window.tot_p)

def get_pd_fir_Ip_hf_in_dt(current_dt):
    # print current_dt
    return (current_dt.fir, current_dt.PD, current_dt.Ip_hf)

#returns a tuple!
def get_transitions_in_dt(current_dt):
    return np.asarray(current_dt[trans_ids])
    
def get_elm_label_in_dt(current_dt):
    return np.asarray([current_dt.sm_elm_label, current_dt.sm_non_elm_label])

def get_state_labels_in_dt(current_dt):
    return np.asarray([current_dt.sm_none_label, current_dt.sm_low_label, current_dt.sm_dither_label, current_dt.sm_high_label])

def get_states_categorical(time_window, num_classes):
    # return to_categorical(time_window.LHD_label.values, num_classes = num_classes)
    categorical = []
    # print(time_window.LHD_label)
    # exit(0)
    int_vals = time_window.LHD_label.values
    for k in range(len(time_window)):
        cat = np.zeros(num_classes)
        cat[int_vals[k]-1] = 1
        # print(cat,int_vals[k]-1)
        categorical.append(cat)
    return np.asarray(categorical)
