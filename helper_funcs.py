import numpy as np
import pandas as pd
import datetime
from label_smoothing import *
from scipy import stats
import pickle
from decimal import *
import copy
import sys

class Path():
    def __init__(self, initial_trans_prob, initial_trans, initial_state): #initial trans, initial state must be lists!
        self.probability = Decimal(float(initial_trans_prob))
        self.transitions = initial_trans
        self.plasma_states = initial_state
        self.length = 1
        # self.uid = np.random.randint(1e8, size=1)
        # print('list', self.plasma_states)
        # exit(0)
        
    def equal_state(self, other):
        if self.plasma_states[-1] == other.plasma_states[-1]:
            return True
        else:
            return False

    def update(self, trans, elm, trans_ids):
        self.length += 1
        pre_upd_prob = self.probability
        pre_upd_trans = self.transitions[:]
        pre_upd_states = self.states[:]
        self.probability = self.probability + Decimal(Trans.no_trans(trans))
        self.transitions.append(Trans.no_trans(trans_ids)) # no transition
        self.states.append(self.states[-1]) #keep last state

        if self.plasma_states[-1] == 1:
            new_path_H = Path(pre_upd_prob + Decimal(Trans.LH(trans)), pre_upd_trans + [Trans.LH(trans_ids)], pre_upd_states + [3])
            # if elm == 0:
            #     return [new_path_H,]
            new_path_D = Path(pre_upd_prob + Decimal(Trans.LD(trans)), pre_upd_trans + [Trans.LD(trans_ids)], pre_upd_states + [2])
            return [self, new_path_H, new_path_D]
        if self.plasma_states[-1] == 2:
            new_path_H = Path(pre_upd_prob + Decimal(Trans.DH(trans)), pre_upd_trans + [Trans.DH(trans_ids)], pre_upd_states + [3])
            new_path_L = Path(pre_upd_prob + Decimal(Trans.DL(trans)), pre_upd_trans + [Trans.DL(trans_ids)], pre_upd_states + [1])
            return [self, new_path_L, new_path_H]
        if self.plasma_states[-1] == 3:
            new_path_L = Path(pre_upd_prob + Decimal(Trans.HL(trans)), pre_upd_trans + [Trans.HL(trans_ids)], pre_upd_states + [1])
            new_path_D = Path(pre_upd_prob + Decimal(Trans.HD(trans)), pre_upd_trans + [Trans.HD(trans_ids)], pre_upd_states + [2])
            return [self, new_path_L, new_path_D]
            
        
    def __str__(self,): #+ str(self.uid)
        return 'path '+ str(self.uid)+ '. prob.:' + str(self.probability)+ '; path size:'+ str(self.length)+ '; last trans/state:'+ str(self.transitions[-1]) + '/'+str(self.plasma_states[-1])
            

class Beam(Path):
    def __init__(self, initial_trans_prob, initial_trans, initial_plasma_state, hidden_states=0, attention_context=0, attention_weights=[], num_outputs=7):
        #initial trans, initial state must be lists!
        self.probability = initial_trans_prob
        self.transitions = np.asarray(initial_trans).astype(np.int32)
        # print(initial_trans)
        self.plasma_states = np.asarray(initial_plasma_state).astype(np.int32)
        self.length = len(initial_trans)
        self.hidden_states = hidden_states
        self.attention_context = attention_context
        self.attention_weights = np.asarray(attention_weights)
        # self.uid = self.uid = np.random.randint(1e8, size=1)[0]
        self.uid = hash(tuple(self.plasma_states))
        # print('list', hash(tuple(self.plasma_states)))
        self.num_outputs = num_outputs
        
    def __eq__(self, other):
        # if self.transitions[]
        #careful here, need cases for state and trans (should add up to 9 cases)
        if self.transitions[-1] == other.transitions[-1] and self.plasma_states[-1] == other.plasma_states[-1]:
            return True
        else:
            return False

    def update(self, log_transition_probs, new_hidden_states, new_attention_context, new_attention_weights):
        assert log_transition_probs.shape[-1] == self.num_outputs
        self.length += 1
        pre_upd_prob = self.probability
        pre_upd_trans = self.transitions[:]
        pre_upd_states = self.plasma_states[:]
        log_transition_probs = np.squeeze(log_transition_probs)
        new_prob = log_transition_probs[Transitions.no_trans()]
        self.probability = self.probability + new_prob
        self.transitions = np.append(self.transitions, Transitions.no_trans()) # no transition
        # print(self.transitions.shape)
        # exit(0)
        self.plasma_states = np.append(self.plasma_states, self.plasma_states[-1])
        self.hidden_states = new_hidden_states
        self.attention_context = new_attention_context
        self.attention_weights = np.append(self.attention_weights, new_attention_weights, axis=0)
        # print(new_attention_weights.shape, self.attention_weights.shape)
        # exit(0)
        if self.plasma_states[-1] == 1:
            LH, LD = Transitions.LH(), Transitions.LD()
            new_beam_H = Beam(pre_upd_prob + log_transition_probs[LH],
                              # pre_upd_trans + [LH],
                              np.append(pre_upd_trans, LH),
                              # pre_upd_states + [States.H()],
                              np.append(pre_upd_states, States.H()),
                              self.hidden_states,
                              self.attention_context,
                              self.attention_weights)
            new_beam_D = Beam(pre_upd_prob +log_transition_probs[LD],
                              # pre_upd_trans + [LD],
                              np.append(pre_upd_trans, LD),
                              # pre_upd_states +[States.D()],
                              np.append(pre_upd_states, States.D()),
                              self.hidden_states,
                              self.attention_context,
                              self.attention_weights)
            return [self, new_beam_H, new_beam_D]
        if self.plasma_states[-1] == 2:
            DH, DL = Transitions.DH(), Transitions.DL()
            new_beam_H = Beam(pre_upd_prob + log_transition_probs[DH],
                              # pre_upd_trans + [DH],
                              np.append(pre_upd_trans, DH),
                              # pre_upd_states + [States.H()],
                              np.append(pre_upd_states, States.H()),
                              self.hidden_states,
                              self.attention_context,
                              self.attention_weights)
            new_beam_L = Beam(pre_upd_prob + log_transition_probs[DL],
                              # pre_upd_trans + [DL],
                              np.append(pre_upd_trans, DL),
                              # pre_upd_states + [States.L()],
                              np.append(pre_upd_states, States.L()),
                              self.hidden_states,
                              self.attention_context,
                              self.attention_weights)
            return [self, new_beam_L, new_beam_H]
        if self.plasma_states[-1] == 3:
            HL,HD = Transitions.HL(), Transitions.HD()
            new_beam_L = Beam(pre_upd_prob + log_transition_probs[HL],
                              # pre_upd_trans + [HL],
                              np.append(pre_upd_trans, HL),
                              # pre_upd_states+ [States.L()],
                              np.append(pre_upd_states, States.L()),
                              self.hidden_states,
                              self.attention_context,
                              self.attention_weights)
            new_beam_D = Beam(pre_upd_prob + log_transition_probs[HD],
                              # pre_upd_trans + [HD],
                              np.append(pre_upd_trans, HD),
                              # pre_upd_states + [States.D()],
                              np.append(pre_upd_states, States.D()),
                              self.hidden_states,
                              self.attention_context,
                              self.attention_weights)
            return [self, new_beam_L, new_beam_D]
            

class StateBeam(Beam):
    def __init__(self, initial_trans_prob, initial_trans, initial_plasma_state, hidden_states=0, attention_context=0, attention_weights=[], num_outputs=3):
        #initial trans, initial state must be lists!
        Beam.__init__(self, initial_trans_prob, initial_trans, initial_plasma_state, hidden_states, attention_context, attention_weights, num_outputs)
        
    def __eq__(self, other):
        # if self.transitions[]
        #careful here, need cases for state and trans (should add up to 9 cases)
        if self.plasma_states[-1] == other.plasma_states[-1]:
            return True
        else:
            return False

    def update(self, log_plasma_state_probs, new_hidden_states, new_attention_context, new_attention_weights):
        # print(log_plasma_state_probs.shape, self.num_outputs)
        assert log_plasma_state_probs.shape[-1] == self.num_outputs
        self.length += 1
        pre_upd_prob = self.probability
        pre_upd_trans = self.transitions[:]
        pre_upd_states = self.plasma_states[:]
        log_transition_probs = np.squeeze(log_plasma_state_probs)
        
        self.hidden_states = new_hidden_states
        self.attention_context = new_attention_context
        if self.length == 1:
            self.attention_weights = new_attention_weights
        else:
            self.attention_weights = np.append(self.attention_weights, new_attention_weights, axis=0)
            
        # print(new_attention_weights.shape)
        # print(self.attention_weights.shape)
        # exit(0)
        if len(pre_upd_states) == 0:
            trans_to_l = Transitions.no_trans()
            trans_to_d = Transitions.no_trans()
            trans_to_h = Transitions.no_trans()
        else:
            if pre_upd_states[-1] == 1:
                trans_to_l = Transitions.no_trans()
                trans_to_d = Transitions.LD()
                trans_to_h = Transitions.LH()
            elif pre_upd_states[-1] == 2:
                trans_to_l = Transitions.DL()
                trans_to_d = Transitions.no_trans()
                trans_to_h = Transitions.DH()
            elif pre_upd_states[-1] == 3:
                trans_to_l = Transitions.HL()
                trans_to_d = Transitions.HD()
                trans_to_h = Transitions.no_trans()
        new_beam_L = StateBeam(pre_upd_prob + log_transition_probs[States.L()-1],
                              np.append(pre_upd_trans, trans_to_l),
                              np.append(pre_upd_states, States.L()),
                              self.hidden_states,
                              self.attention_context,
                              self.attention_weights)
        new_beam_H = StateBeam(pre_upd_prob + log_transition_probs[States.H()-1],
                               np.append(pre_upd_trans, trans_to_h),
                              np.append(pre_upd_states, States.H()),
                              self.hidden_states,
                              self.attention_context,
                              self.attention_weights)
        new_beam_D = StateBeam(pre_upd_prob +log_transition_probs[States.D()-1],
                               np.append(pre_upd_trans, trans_to_d),
                              np.append(pre_upd_states, States.D()),
                              self.hidden_states,
                              self.attention_context,
                              self.attention_weights)
        return [new_beam_L, new_beam_D, new_beam_H]

class Trans():
    @staticmethod
    def LH(trans):
        return trans[0]
    @staticmethod
    def HL(trans):
        return trans[1]
    @staticmethod
    def DH(trans):
        return trans[2]
    @staticmethod
    def HD(trans):
        return trans[3]
    @staticmethod
    def LD(trans):
        return trans[4]
    @staticmethod
    def DL(trans):
        return trans[5]
    @staticmethod
    def no_trans(trans):
        return trans[-1]

class Transitions():
    @staticmethod
    def trans_ids():
        return [0,1,2,3,4,5,6]
    @staticmethod
    def LH():
        return Transitions.trans_ids()[0]
    @staticmethod
    def HL():
        return Transitions.trans_ids()[1]
    @staticmethod
    def DH():
        return Transitions.trans_ids()[2]
    @staticmethod
    def HD():
        return Transitions.trans_ids()[3]
    @staticmethod
    def LD():
        return Transitions.trans_ids()[4]
    @staticmethod
    def DL():
        return Transitions.trans_ids()[5]
    @staticmethod
    def no_trans():
        return Transitions.trans_ids()[-1]

class States():
    @staticmethod
    def states_ids():
        return [0,1,2,3]
    @staticmethod
    def N():
        return States.states_ids()[0]
    @staticmethod
    def L():
        return States.states_ids()[1]
    @staticmethod
    def D():
        return States.states_ids()[2]
    @staticmethod
    def H():
        return States.states_ids()[3]
    
#class to keep track of paths which have the same last transition and state
# class PathHeap():
#     def __init__(self, path):
#         self.paths = [path,]
#     
    
def viterbi_search(transitions, elms):
    #trans should have shape (shot_length, 7)
    trans_ids = ['LH', 'HL', 'DH', 'HD', 'LD', 'DL', 'no_trans']
    trans_ids = [0,1,2,3,4,5,6]
    states = [1, 2, 3]
    print('in viterbi search')
    # initial_trans = np.log(transitions[0]) #['LH', 'HL', 'DH', 'HD', 'LD', 'DL', 'no_trans']
    initial_trans = np.zeros(transitions.shape[1]) + np.finfo(float).eps
    # print(initial_trans)
    initial_trans[Trans.no_trans(trans_ids)] = 1 - np.finfo(float).eps*(len(initial_trans)-1)
    # print(initial_trans, sum(initial_trans))
    initial_trans = np.log(initial_trans)
    
    #initial path MUST start in L mode
    
    paths = [Path(Trans.no_trans(initial_trans), [Trans.no_trans(trans_ids),], [1,]),] #starts in Low (code = 1)
    
    for k in range(1, len(transitions)):
        trans = np.log(transitions[k]).tolist()
        elm = np.log(np.clip(elms[k], a_min=np.finfo(float).eps, a_max=None)).tolist()
        paths_temp = []
        # print(k, 'no_paths', len(paths))
        for path in paths:
            # print('path ', str(path), ' ' , end='')
            paths_temp.extend(path.update(trans, elm, trans_ids))
        paths_aux = []
        
        buckets = {state:[] for state in states}
        # print('buckets', buckets)
        for path_temp in paths_temp:
            # print('path_temp', path_temp)
            buckets[path_temp.states[-1]].append(path_temp)
        # print('buckets', buckets)
        for paths in buckets.values():
            if len(paths) == 0:
                continue
            paths_aux.append(max_prob_path(paths))
                
                
            # paths_temp.pop[0]
        paths = paths_aux
        # mp = max_prob_path(paths)
        # print('mp', mp.probability)
        
    path_max_prob = max_prob_path(paths)
    # print(path_max_prob.states)
    
    cat_transitions = []
    max_prob_trans = np.asarray(path_max_prob.transitions)
    # print(max_prob_trans.shape)
    for k in range(len(max_prob_trans)):
        t = max_prob_trans[k]
        cat_transitions.append(np.zeros(7))
        cat_transitions[-1][t] = 1
    cat_transitions = np.asarray(cat_transitions)
    print('path of maximum probability:', path_max_prob.probability, path_max_prob.probability.exp())
    return np.asarray(path_max_prob.states), cat_transitions


#if two paths are moving into the same state, they will be equal from there on
#thus, we only need to find the one with the highest probability until the point in time where they both move into the same state
def max_prob_path(paths):
    # prob = 0
    path_max_prob = paths[0]
    prob = path_max_prob.probability
    for path in paths[1:]:
        if path.probability > prob:
            prob = path.probability
            path_max_prob = path
    return path_max_prob



def get_date_time_formatted():
    dtime = datetime.datetime.now()
    year = dtime.year
    month = dtime.month
    day = dtime.day
    hour = dtime.hour
    mi = dtime.minute
    sec = dtime.second
    return str(day) + '-' + str(month) + '-' + str(year) + '-' + str(hour) + ':'+ str(mi) + ':'+ str(sec)

#Expects a pandas Dataframe with current in A
def normalize_current_MA(df):
    df['IP'] = df['IP'].divide(1e6)
    return df

def remove_disruption_points(df):
    pd_vals = df.PD.values
    k = -1
    pd1 = pd_vals[k]
    pd2 = pd_vals[k-1]
    while pd2 == pd1:
        k -= 1
        pd1 = pd_vals[k]
        pd2 = pd_vals[k-1]    
    no_disr_shot = df.iloc[:(k - 200)]
    # if k < -1:
    #     print('removed disruption from this shot.')# original shot len', len(df), 'new shot len', len(no_disr_shot))
    return no_disr_shot
                      
#Expects a pandas Dataframe with current in A
def remove_current_30kA(df):
    return df[df['IP'].abs() > 30*1e3]

def remove_current_400kA(df):
    return df[df['IP'].abs() > 400*1e3]

def remove_current_150kA(df):
    return df[df['IP'].abs() > 150*1e3]

def remove_current_370kA(df):
    return df[df['IP'].abs() > 370*1e3]

def remove_current_750kA(df):
    return df[df['IP'].abs() > 750*1e3]

def remove_current_1MA(df):
    return df[df['IP'].abs() > 1000*1e3]

def remove_current_50kA(df):
    return df[df['IP'].abs() > 50*1e3]

def remove_no_state(df):
    return df[df['LHD_label'] != 0]

def remove_conv_start_end(df):
    return df[20 : len(df)-20]

def replace_NaNs(df):
    df['LHD_label'] = df['LHD_label'].fillna(0)
    df['PD'] = df['PD'].fillna(0)
    try: 
        df['WP'] = df['WP'].fillna(0)
    except KeyError:
        print('Could not find WP key!')
    try:
        df['GWfr'] = df['GWfr'].fillna(0)
    except KeyError:
        print('Could not find GWfr key!')
    return df

def removeNans(df):
    # print(df)
    mask = np.zeros(len(df)).astype(np.bool)
    for column in ['PD', 'IP', 'DML', 'FIR', 'LHD_label']:
        nans = df[column].isnull().values
        mask = np.logical_or(mask, nans)
        # try:
        #     nans = df[column].isnull().values
        #     # print(nans)
        #     print(np.where(nans==False))
        # except:
        #     print("Unexpected error:", sys.exc_info()[0])
        #     raise
    # print(mask.shape)
    # print(np.sum(mask))
    df_filtered = df[np.logical_not(mask)].copy()
    # print(df_filtered)
    # exit(0)
    return df_filtered

def normalize_signals_mean(df, machine_id, func=''):
    if func=='minmax':
        #df['PD'] = (df.PD.values - np.min( df.PD.values))/(np.max(df.PD.values) -  np.min(df.PD.values))
        #df['FIR'] = (df.FIR.values - np.min( df.FIR.values))/(np.max(df.FIR.values) -  np.min(df.FIR.values))
        df['PD'] = df.PD.values/np.max(df.PD.values)
        df['FIR'] = df.FIR.values/np.max(df.FIR.values)
        try:
            df['WP'] = (df.WP.values - np.min( df.WP.values))/(np.max(df.WP.values) -  np.min(df.WP.values))
        except AttributeError:
            df['DML'] = (df.DML.values - np.min( df.DML.values))/(np.max(df.DML.values) -  np.min(df.DML.values))
    elif func=='avg':
        ### Actual values used in real-time
        if machine_id == 'TCV':
            norm_fir = 1e-19
            norm_pd = 1.0
        elif machine_id == 'JET':
            norm_fir = 1e-20
            norm_pd = 1e-14

        df['FIR'] = df.FIR.values*norm_fir
        df['PD'] = df.PD.values*norm_pd

    elif func=='avg_GCS': # Average normalization for generalized confinement state detector
        if machine_id == 'TCV': # obtained from TCV train set
            norm_wp = 9293
            norm_fir = 0.086 # GWfr
            norm_pd = 0.001
        elif machine_id == 'JET':
            norm_wp = 2.5e+6
            norm_fir = 0.33
            norm_pd = 5.6e+10
        df['GWfr'] = df.GWfr.values/norm_fir
        df['PD'] = df.PD.values/norm_pd
        df['WP'] = df.WP.values/norm_wp

    elif func=='z_GCS': # Z-score normalization for generalized confinement state detector
        if machine_id == 'TCV': # obtained from TCV train set
            mu_wp = 1.1e+4
            sigma_wp = 1e+4
            mu_gwfr = 0.15
            sigma_gwfr = 0.15
            #mu_fir = 3.5e+19
            #sigma_fir = 2.5e+19
            mu_pd = 0
            sigma_pd = 4
        elif machine_id == 'JET':
            mu_wp = 7e+6
            #mu_wp = 0
            sigma_wp = 6e+6
            #sigma_wp = 1
            mu_gwfr = 0.5
            #mu_gwfr = 0
            sigma_gwfr = 0.5
            #sigma_gwfr = 1
            ##mu_fir = 5.6e+19
            ##sigma_fir = 5.6e+19
            mu_pd = 0
            #mu_pd = 0
            sigma_pd = 1e+14
            #sigma_pd = 1
        elif machine_id == 'AUG':
            mu_wp = 6e+5
            sigma_wp = 3e+5
            mu_gwfr = 0.45
            sigma_gwfr = 0.6
            #mu_fir = 6e+19
            #sigma_fir = 4e+19
            mu_pd = 0
            sigma_pd = 1
        df['GWfr'] = (df.GWfr.values-mu_gwfr)/sigma_gwfr
        #df['FIR'] = (df.FIR.values-mu_fir)/sigma_fir
        df['PD'] = (df.PD.values-mu_pd)/sigma_pd
        df['WP'] = (df.WP.values-mu_wp)/sigma_wp
    return df

def det_trans_to_state(shot_tseries_trans):
    lhd_det = []
    state = 1 # L
    trans_ids = [tr + '_det' for tr in get_trans_ids()]
    # print('here', trans_ids)
    # print(shot_tseries_trans)
    for ind, t in shot_tseries_trans.iterrows():
        # current_t = shot_tseries_trans[t]
        if t['LH_det'] == 1 and state == 1:
            state = 3
        if t['DH_det'] == 1 and state == 2:
            state = 3
        if t['LD_det'] == 1 and state == 1:
            state = 2
        if t['HD_det'] == 1 and state == 3:
            state = 2
        if t['HL_det'] == 1 and state == 3:
            state = 1
        if t['DL_det'] == 1 and state == 2:
            state = 1
            # print('state change @', t.time)
        lhd_det.append(state)
    lhd_det = np.asarray(lhd_det)
    # print(lhd_det.shape)
    # print(lhd_det)
    # exit(0)
    return lhd_det


def event_cont_to_disc_sev(events_cont, threshold = .5, gaussian_hinterval=10):
    return elms_cont_to_disc_sev(events_cont, threshold, gaussian_hinterval)

def event_cont_to_disc(events_cont, threshold = .5, gaussian_hinterval=10):
    return elms_cont_to_disc(events_cont, threshold, gaussian_hinterval)

#converts a continuous elm sequence to a discrete one.
#assumes that consecutive elms can happen immediately after each other, i.e., separated by as little as on time slice. 
def elms_cont_to_disc_sev(elms_cont, threshold = .5, gaussian_hinterval=10):
    # index of the model output where an elm is detected
    cut = np.argwhere(elms_cont >= threshold)
    # print('discretizing elms in variable intervals...')
    cut_ids = np.arange(len(elms_cont))[cut][:,0] # TODO check if it should be done with np.where
    cut_p = elms_cont[cut][:,0]
    
    # sort the model outputs
    # sorted indexes
    sort = np.argsort(cut_p)
    sorted_p = cut_p[sort]
    # print('elms_cont', elms_cont.shape)
    sorted_ids = cut_ids[sort]
    sorted_vals = np.asarray([sorted_p, sorted_ids]).swapaxes(0,1)
    deltas = []
    detected_peaks = 0
    # print('lencut', len(cut), len(elms_cont), threshold)
    # print('cut ids', cut_ids)
    # n = 10
    while(len(sorted_vals > 0)):
        # print('-------------------------len(sorted_vals)--------------------------------', sorted(sorted_vals[:,1]), len(sorted_vals))
        current_id = sorted_vals[-1, 1] #start from the end of sorted_vals nparray
        current_val = sorted_vals[-1, 0]
        # print(current_val, current_id)
        deltas += [current_id.astype(int)]
        # for each peak in the elm outputs of the cnn,
        # we have to check if the values to its left are strictly increasing, and the ones to the right strictly decreasing.
        #If so, we remove all of them. If not, we remove the interval where the aforementioned condition is met.
        #The remaining points will correspond to another peak, which is close by, and which may or not be a true positive. 
        left_of_peak = np.linspace(current_id - gaussian_hinterval, current_id, num=gaussian_hinterval+1, endpoint=True).astype(int)
        # print('left_of_peak', left_of_peak)
        # print('in1d', np.in1d(cut_ids, left_of_peak))
        # print('cut ids', cut_ids)
        right_of_peak = np.linspace(current_id, current_id+gaussian_hinterval, num=gaussian_hinterval+1, endpoint=True).astype(int)
        left_values = cut_p[np.where(np.in1d(cut_ids, left_of_peak))[0]] #which of the ids in left of peak are in the ids stored in sorted vals? get their ids, and then their values
        # print('left_values', left_values)
        min_left_val = min_left(left_values)
        left_to_remove = len(left_values[min_left_val:])
        right_values = cut_p[np.where(np.in1d(cut_ids, right_of_peak))[0]]
        # print('right_values', right_values)
        min_right_val = min_right(right_values)
        right_to_remove = len(right_values[:min_right_val])
        # print('left_to_remove', left_to_remove, 'right_to_remove', right_to_remove)
        around = np.linspace(current_id - left_to_remove, current_id + right_to_remove, num=left_to_remove+right_to_remove+1, endpoint=True).astype(int)
        # print('around', around)
        ids_to_remove = np.where(np.in1d(sorted_vals[:, 1], around))[0] #remove all ids around this positive network output
        # print('ids_to_remove', ids_to_remove)
        sorted_vals = np.delete(sorted_vals, ids_to_remove, axis=0)
        detected_peaks += 1
        # print(detected_peaks)
    delta_mask = np.zeros(len(elms_cont))
    delta_mask[deltas] = 1
    return delta_mask


#converts a continuous elm sequence to a discrete one.
#assumes that consecutive elms can only happen every 2*gaussian_hinterval time slices. 
def elms_cont_to_disc(elms_cont, threshold = .5, gaussian_hinterval=10):
    # index of the model output where an elm is detected
    cut = np.argwhere(elms_cont >= threshold)
    # print('discretizing elms in fixed interval...')
    cut_ids = np.arange(len(elms_cont))[cut][:,0] # TODO check if it should be done with np.where
    cut_p = elms_cont[cut][:,0]
    
    # sort the model outputs
    # sorted indexes
    sort = np.argsort(cut_p)
    sorted_p = cut_p[sort]
    # print('elms_cont', elms_cont.shape)
    sorted_ids = cut_ids[sort]
    sorted_vals = np.asarray([sorted_p, sorted_ids]).swapaxes(0,1)
    deltas = []
    detected_peaks = 0
    # print('lencut', len(cut), len(elms_cont), threshold)
    # print('cut ids', cut_ids)
    # n = 10
    while(len(sorted_vals > 0)):
        # print('-------------------------len(sorted_vals)--------------------------------', sorted(sorted_vals[:,1]), len(sorted_vals))
        current_id = sorted_vals[-1, 1] #start from the end of sorted_vals nparray
        current_val = sorted_vals[-1, 0]
        # print(current_val, current_id)
        deltas += [current_id.astype(int)]
        # for each peak in the elm outputs of the cnn,
        # we have to check if the values to its left are strictly increasing, and the ones to the right strictly decreasing.
        #If so, we remove all of them. If not, we remove the interval where the aforementioned condition is met.
        #The remaining points will correspond to another peak, which is close by, and which may or not be a true positive. 
        left_of_peak = np.linspace(current_id - gaussian_hinterval, current_id, num=gaussian_hinterval+1, endpoint=True).astype(int)
        # print('left_of_peak', left_of_peak)
        # print('in1d', np.in1d(cut_ids, left_of_peak))
        # print('cut ids', cut_ids)
        right_of_peak = np.linspace(current_id, current_id+gaussian_hinterval, num=gaussian_hinterval+1, endpoint=True).astype(int)
        left_values = cut_p[np.where(np.in1d(cut_ids, left_of_peak))[0]] #which of the ids in left of peak are in the ids stored in sorted vals? get their ids, and then their values
        # print('left_values', left_values)
        min_left_val = min_left(left_values)
        # min_left_val = -n
        left_to_remove = len(left_values[min_left_val:])
        right_values = cut_p[np.where(np.in1d(cut_ids, right_of_peak))[0]]
        # print('right_values', right_values)
        min_right_val = min_right(right_values)
        # min_right_val = n
        right_to_remove = len(right_values[:min_right_val])
        # print('left_to_remove', left_to_remove, 'right_to_remove', right_to_remove)
        around = np.linspace(current_id - gaussian_hinterval, current_id + gaussian_hinterval, num=2*gaussian_hinterval+1, endpoint=True).astype(int)
        # print('around', around)
        ids_to_remove = np.where(np.in1d(sorted_vals[:, 1], around))[0] #remove all ids around this positive network output
        # print('ids_to_remove', ids_to_remove)
        sorted_vals = np.delete(sorted_vals, ids_to_remove, axis=0)
        detected_peaks += 1
        # print(detected_peaks)
    delta_mask = np.zeros(len(elms_cont))
    delta_mask[deltas] = 1
    return delta_mask

def dice_coefficient(predicted_states, labeled_states):
    total_low_intersects = 0
    total_low_state_trues = 0
    total_low_state_positives = 0
    total_high_intersects = 0
    total_high_state_trues = 0
    total_high_state_positives = 0
    total_dither_intersects = 0
    total_dither_state_trues = 0
    total_dither_state_positives = 0
    # predicted_states += 1 #no none state
    # print(len(predicted_states), len(labeled_states))
    assert(len(predicted_states) == len(labeled_states))
    
    # none_state_positives = np.zeros(len(predicted_states))
    # none_state_positives[predicted_states == 0] = 1
    low_state_positives = np.zeros(len(predicted_states))
    low_state_positives[predicted_states == 1] = 1
    # print('here', np.sum(low_state_positives))
    dither_state_positives = np.zeros(len(predicted_states))
    dither_state_positives[predicted_states == 2] = 1
    high_state_positives = np.zeros(len(predicted_states))
    high_state_positives[predicted_states == 3] = 1
    # a,b,c,d = np.sum(none_state_positives),
    a,b,c, = np.sum(low_state_positives), np.sum(dither_state_positives), np.sum(high_state_positives)
    # print('posit', a, b, c, a+b+c)
    
    # none_state_trues = np.zeros(len(labeled_states))
    # none_state_trues[labeled_states == 0] = 1   
    low_state_trues = np.zeros(len(labeled_states))
    low_state_trues[labeled_states == 1] = 1
    # plt.plot(labeled_states)
    # plt.show()
    dither_state_trues = np.zeros(len(labeled_states))
    dither_state_trues[labeled_states == 2] = 1
    high_state_trues = np.zeros(len(labeled_states))
    high_state_trues[labeled_states == 3] = 1
    # print(sum(dither_state_trues))
    # exit(0)
    # a,b,c,d = np.sum(none_state_trues),
    a,b,c = np.sum(low_state_trues), np.sum(dither_state_trues), np.sum(high_state_trues)
    # print('trues',  a, b, c, a+b+c)
    
    # none_intersect_cardinality = np.sum(np.logical_and(none_state_positives, none_state_trues))
    # total_none_intersects += none_intersect_cardinality
    # total_none_state_trues += np.sum(none_state_trues)
    # total_none_state_positives += np.sum(none_state_positives)  
    
    low_intersect_cardinality = np.sum(np.logical_and(low_state_positives, low_state_trues))
    total_low_intersects += low_intersect_cardinality
    total_low_state_trues += np.sum(low_state_trues)
    total_low_state_positives += np.sum(low_state_positives)
    # low_dsc = (2.*low_intersect_cardinality)/(np.sum(low_state_trues) + np.sum(low_state_positives))
    
    dither_intersect_cardinality = np.sum(np.logical_and(dither_state_positives, dither_state_trues))
    total_dither_intersects += dither_intersect_cardinality
    total_dither_state_trues += np.sum(dither_state_trues)
    total_dither_state_positives += np.sum(dither_state_positives)
    # print np.sum(intersect_dither)
    # dither_dsc = (2.*dither_intersect_cardinality)/(np.sum(dither_state_trues) + np.sum(dither_state_positives))
    
    high_intersect_cardinality = np.sum(np.logical_and(high_state_positives, high_state_trues))
    total_high_intersects += high_intersect_cardinality
    total_high_state_trues += np.sum(high_state_trues)
    total_high_state_positives += np.sum(high_state_positives)
    # high_dsc = (2.*high_intersect_cardinality)/(np.sum(high_state_trues) + np.sum(high_state_positives))
    
    # a,b,c, d= total_none_state_positives, total_low_state_positives, total_dither_state_positives, total_high_state_positives
    a,b,c = total_low_state_positives, total_dither_state_positives, total_high_state_positives
    # print('positives', a, b, c, a+b+c)
    # a,b,c, d= none_intersect_cardinality, low_intersect_cardinality, dither_intersect_cardinality, high_intersect_cardinality
    a,b,c = low_intersect_cardinality, dither_intersect_cardinality, high_intersect_cardinality
    # print('itsct', a, b, c, a+b+c)
    
    # if(total_none_state_trues + total_none_state_positives) > 0:
    #     none_dsc = (2.*total_none_intersects)/(total_none_state_trues + total_none_state_positives)
    # else:
    #     none_dsc = 1
    if(total_low_state_trues + total_low_state_positives) > 0:
        low_dsc = (2.*total_low_intersects)/(total_low_state_trues + total_low_state_positives)
    else:
        low_dsc = 1
    if(total_dither_state_trues + total_dither_state_positives) > 0:
        dither_dsc = (2.*total_dither_intersects)/(total_dither_state_trues + total_dither_state_positives)
    else:
        # print('dither dsc')
        # print(total_dither_state_trues)
        # print(total_dither_state_positives)
        dither_dsc = 1
    if(total_high_state_trues + total_high_state_positives) > 0:
        high_dsc = (2.*total_high_intersects)/(total_high_state_trues + total_high_state_positives)
    else:
        high_dsc = 1
    
    # s_nst = sum(none_state_trues)
    s_lst = sum(low_state_trues)
    s_hst = sum(high_state_trues)
    s_dst = sum(dither_state_trues)
    # none_state_trues_pc = s_nst/len(labeled_states)
    low_state_trues_pc = s_lst/len(labeled_states)
    high_state_trues_pc = s_hst/len(labeled_states)
    dither_state_trues_pc = s_dst/len(labeled_states)
    # total_dsc = none_dsc * none_state_trues_pc + low_dsc*low_state_trues_pc + high_dsc*high_state_trues_pc + dither_dsc*dither_state_trues_pc
    
    total_dsc = low_dsc*low_state_trues_pc + high_dsc*high_state_trues_pc + dither_dsc*dither_state_trues_pc
    # print('Calc of mean val for dice', low_dsc, low_state_trues_pc, high_dsc, high_state_trues_pc, dither_dsc, dither_state_trues_pc, total_dsc)
    return np.asarray([low_dsc, dither_dsc, high_dsc, total_dsc])
    

def k_statistic(predicted, labeled):
    k_index = []
    state_trues_pc = []
    for i, state in enumerate(['L', 'D', 'H']):
        s = i + 1
        assert(len(predicted) == len(labeled))
        predicted_states = np.zeros(len(predicted))
        predicted_states[predicted == s] = 1 
        labeled_states = np.zeros(len(labeled))
        labeled_states[labeled == s] = 1
        
        # if np.array_equal(labeled_states, np.zeros(len(labeled))) and sum(predicted_states) < 2:
        #     predicted_states = np.zeros(len(predicted))
        # 
        
        s_tot = sum(labeled_states)
        state_trues_pc += [round(s_tot/len(labeled_states),3)]
        yy = 0
        yn = 0
        ny = 0
        nn = 0
        total = len(predicted_states)
        for k in range(total):
            if predicted_states[k] == 1 and labeled_states[k] == 1:
                yy += 1
            elif predicted_states[k] == 0 and labeled_states[k] == 0:
                nn += 1
            elif predicted_states[k] == 1 and labeled_states[k] == 0:
                ny += 1
            elif predicted_states[k] == 0 and labeled_states[k] == 1:
                yn += 1
        assert(yy + yn + ny + nn == total)
        p0 = (yy + nn) /  total
        pyes = ((yy + yn) / total) * ((yy + ny) / total)
        pno = ((ny + nn) / total) * ((yn + nn) / total)
        pe = pyes + pno
        # print('ps....................', p0, pyes, pno, pe)
        if pe == 1:
            k_index +=[1]
        else:
            score = (p0 - pe) / (1-pe)
            if score < 0: #careful, as k-statistic can actually be smaller than 0!
                score = 0
            k_index += [score]
    mean_k_ind = np.average(np.asarray(k_index), weights=state_trues_pc)
    #print('state_trues_pc ', state_trues_pc)
    return np.asarray(k_index + [mean_k_ind])

def confusion_matrix_blocks(predictions, labels, num_classes=3):
    # print(predictions.shape, labels.shape)
    matrix = np.zeros((num_classes, num_classes)).astype(np.int16)
    matrix_order=['predictions', 'labels']
    matrix_order=[0,1]
    for k in range(len(predictions)):
        if labels[k] == -1: #no agreement among labelers, so we skip this timestep
            continue
        else:
            matrix[labels[k]-1, predictions[k]-1] += 1
    # print(matrix[0,0])
    # print(matrix[0,1])
    # print(matrix[2,0])
    # matrix = matrix.astype('|S4')        
    # matrix[0] = ['\\', 'L', 'D', 'H' ]
    # matrix[1,0] = 'L'
    # matrix[2,0] = 'D'
    # matrix[3,0] = 'H'
    # print(matrix)
    # exit(0)
    # plt.matshow(matrix)
    # plt.show()
    return matrix, matrix_order
        
    # exit(0)


def min_left(left_values):
    if len(left_values) == 1:
        return 0
    k = -1
    start = left_values[k]
    while(left_values[k-1]<=left_values[k]):
        k -= 1
        if abs(k) == len(left_values):
            return k+1
    return k + 2

def min_right(right_values):
    if len(right_values) == 1:
        return 0
    k = 0
    start = right_values[k]
    while(right_values[k+1]<right_values[k]):
        k += 1
        if k == len(right_values)-1:
            break
    return k

#Receives: array of labels and predictions for any event (ELM or transition). Incoming arrays are binary (only 0s and 1s).
#Receives: optional window in which the function will consider a prediction to be correct (default is 10 wide)
#Receives: optional argument with times for one hot event labels equal to 1, for debugging purposes
def conf_matrix(predicted_events, labeled_events, gaussian_hinterval=10, signal_times=[]):
    return elm_conf_matrix(predicted_events, labeled_events, gaussian_hinterval, signal_times)


#Receives: array of labels and predictions for ELMs. Incoming arrays are binary (only 0s and 1s).
#Receives: optional window in which the function will consider a prediction to be correct (default is 10 wide)
#Receives: optional argument with times for one hot elm labels equal to 1, for debugging purposes
def elm_conf_matrix(predicted_elms, labeled_elms, gaussian_hinterval=10, signal_times=[]):
    times = signal_times
    false_pos_location = []
    false_neg_location = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    first_ids = np.arange(-gaussian_hinterval, 0)
    last_ids = np.arange(len(labeled_elms), len(labeled_elms)+gaussian_hinterval)
    one_hot_label_ids = np.where(labeled_elms == 1)[0]
    smooth_labels = smoothen_elm_values(labeled_elms, smooth_window_hsize=gaussian_hinterval)[0].values
    
    smooth_temp = np.copy(smooth_labels)
    # plt.plot(smooth_temp)
    # plt.plot(labeled_elms)
    # plt.show()
    assert len(predicted_elms) == len(labeled_elms)
    delta_mask = np.copy(predicted_elms)
    removed_id_counter = 0
    # ids_removed = []
    
    
    for e_ind, e in enumerate(one_hot_label_ids):
        e_adj = e - removed_id_counter
        left_of_peak = np.linspace(e - gaussian_hinterval, e, num=gaussian_hinterval+1, endpoint=True).astype(int)
        
        # ensure we're not looking for events outside of shot index range
        intersect_ids = sorted(set(left_of_peak) & set(first_ids))
        if(len(intersect_ids)>0):
            left_of_peak = left_of_peak[len(intersect_ids):]
        right_of_peak = np.linspace(e, e + gaussian_hinterval, num=gaussian_hinterval+1, endpoint=True).astype(int)
        intersect_ids = sorted(set(right_of_peak) & set(last_ids))
        # print('new')
        # print(right_of_peak)
        # print(intersect_ids)
        # print(last_ids)
        if(len(intersect_ids)>0):
            right_of_peak = right_of_peak[:-(len(intersect_ids))]
        # print(right_of_peak)    
        left_values = smooth_labels[left_of_peak]
        min_left_val = min_left(left_values)
        right_values = smooth_labels[right_of_peak]
        # print(right_values)
        min_right_val = min_right(right_values)
        # print('min_right_val', min_right_val)
        ids_around = np.linspace(e_adj + min_left_val, e_adj + min_right_val, num=np.abs(min_left_val)+min_right_val+1, endpoint=True).astype(int)
        # print(ids_around)
        # print(len(delta_mask))
        deltas_in_smooth = np.sum(delta_mask[ids_around])
        if deltas_in_smooth == 0:
            false_negatives += 1
            if signal_times != []:
                false_neg_location += [times[ids_around]]
        elif deltas_in_smooth > 1:
            true_positives += 1
            false_positives += deltas_in_smooth - 1
            if signal_times != []:
                false_pos_location += [times[ids_around]]
        else:
            true_positives += 1
        delta_mask = np.delete(delta_mask, ids_around, axis=0)
        smooth_temp = np.delete(smooth_temp, ids_around)
        if signal_times != []:
            times = np.delete(times, ids_around, axis=0)
        removed_id_counter += len(ids_around)
        # ids_removed.append([ids_around])
        
    # false_positives += np.count_nonzero(delta_mask)
    # leftover_ids = sorted(np.where(delta_mask == 1))
    # print(leftover_ids)
    # for leftover_id in leftover_ids:
    #     ids_around = np.linspace(leftover_id - 10, leftover_id + 10, num=21, endpoint=True).astype(int)
    #     delta_mask = np.delete(delta_mask, ids_around, axis=0)
    # ids_removed = sorted(ids_removed)
    fps_to_remove = set()
    # print(len(delta_mask))
    # for p in range(len(delta_mask)):
    d_m = np.arange(len(delta_mask))[delta_mask == 1]
    # print(delta_mask)
    for p in d_m:
        ids_around = np.linspace(p - gaussian_hinterval, p + gaussian_hinterval, num=gaussian_hinterval*2+1, endpoint=True).astype(int)
        ids_around = ids_around[ids_around>=0]
        ids_around = ids_around[ids_around<len(delta_mask)]
        false_positives += 1
        fps_to_remove.update(ids_around)
        # print(ids_around, delta_mask[ids_around], np.any(delta_mask[ids_around] == 1))
        # if np.any(delta_mask[ids_around] == 1):
        #     # fps_to_remove.append([p])
        #     false_positives += 1
        #     print('h')
        # else:
        #     # temp = np.linspace(p - 15, p + 15, num=31, endpoint=True).astype(int)
        #     # 
        #     # print('false one', p)
        #     # print(temp, delta_mask[temp])
        #     # print(np.any(delta_mask[temp] == 1))
        #     pass
    # plt.plot(delta_mask)
    # plt.plot(smooth_temp)
    # plt.show()
    # print(delta_mask, list(fps_to_remove))
    delta_mask = np.delete(delta_mask, list(fps_to_remove))
    # print(delta_mask)
    
    
    
    # if signal_times != []:
    #     fptime = times[np.argwhere(delta_mask!=0)]
    #     if len(fptime) != 0:
    #         false_pos_location += [fptime[:,0]]
    
    true_negatives += np.count_nonzero(delta_mask==0)//(gaussian_hinterval*2)
    assert(false_negatives + true_positives) == len(one_hot_label_ids)
    # exit(0)
    return (true_positives, false_positives, true_negatives, false_negatives)

def conf_metrics(true_positives, false_positives, true_negatives, false_negatives):
    # print(true_positives, false_positives, true_negatives, false_negatives)
    # with np.errstate(divide='ignore'):
    if true_positives == 0:
        tpr = 0
        ppv = 0
    else:
        tpr = np.divide(float(true_positives), true_positives + false_negatives)
        ppv = np.divide(float(true_positives), true_positives + false_positives)
    if true_negatives == 0:
        npv = 0
        spc = 0
    else:
        npv = np.divide(float(true_negatives), true_negatives + false_negatives)
        spc = np.divide(float(true_negatives), true_negatives + false_positives)     
    fpr = 1. - spc
    fnr = 1. - tpr
    fdr = 1 - ppv
    # print()
    return (round(tpr,4), round(spc,4), round(ppv,4), round(npv,4), round(fpr,4), round(fnr,4), round(fdr,4))

def get_roc_curve(pred_elms_continuous, labeled_elms, thresholds, gaussian_hinterval=10, signal_times=[]):
    assert len(pred_elms_continuous) == len(labeled_elms)
    roc_points = {}
    print('computing roc curve...')
    for threshold in thresholds:
        print('roc threshold', threshold)
        sys.stdout.flush()
        pred_elms_disc = elms_cont_to_disc(pred_elms_continuous, threshold = threshold)
        assert len(pred_elms_disc) == len(labeled_elms)
        true_positives, false_positives, true_negatives, false_negatives = elm_conf_matrix(pred_elms_disc, labeled_elms, gaussian_hinterval=gaussian_hinterval, signal_times=[])
        tpr, spc, ppv, npv, fpr, fnr, fdr = conf_metrics(true_positives, false_positives, true_negatives, false_negatives)
        roc_points[threshold] = [fpr, tpr]
    return roc_points

def point_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def youden_index(roc_point):
    return roc_point[1] - roc_point[0]

def get_roc_best(roc_curve):
    best = (0, 1) #fpr = 0, tpr =1
    youden_indexes = {}
    # print(roc_curve)
    for key in roc_curve.keys():
        # d = point_dist(best, roc_curve[key])
        d = youden_index(roc_curve[key])
        youden_indexes[key] = d
    # k, v = min(distances.items(), key=lambda x: x[1])
    k, v = max(youden_indexes.items(), key=lambda x: x[1])
    # print('distances', distances)
    return youden_indexes, k, v

#removes no state, small currents; normalizes current (divides it be 1 MA); removes disruption points.
#Does NOT normalize other shot signal values!!!
def load_fshot_from_labeler(shot_id, machine_id, data_dir):
    #print(shot_id)
    shot, labeler = shot_id.split('-')
    
    fshot = pd.read_csv(data_dir + labeler + '/' + machine_id + '_'  + str(shot) + '_' + labeler + '_labeled.csv')
    if machine_id == 'JET':
        fshot = remove_current_1MA(fshot)
    elif machine_id == 'TCV' or machine_id == 'TCVandAUG':
        fshot = remove_current_30kA(fshot)
    elif machine_id == 'AUG':
        fshot = remove_current_370kA(fshot)
    else:
        raise ValueError('Machine_id {} not stored in the database'.format(machine_id))
    
    fshot = replace_NaNs(fshot)
    fshot = remove_no_state(fshot)
    fshot.LHD_label = [int(i) for i in fshot.LHD_label.values]
    fshot = fshot.reset_index(drop=True)
    
    fshot = normalize_current_MA(fshot)
    fshot = fshot.reset_index(drop=True)
    fshot = remove_disruption_points(fshot)#Remove last 200 data points
    return fshot, fshot.time.values

def load_fshot_from_number(shot_num, machine_id, data_dir, labelers):
    # labeler= 'ffelici' #could be another labeler as well, it doesnt matter since we only want the signals.
    for labeler in labelers:
        # try:
        temp = load_fshot_from_labeler(str(shot_num)+'-'+labeler, machine_id, data_dir)
            # break
        # except:
            # continue
    
    return temp

def load_fshot_from_classifier(shot_id, data_dir):  #, machine_id
    shot, labeler = shot_id.split('-')
    print(shot, labeler, data_dir + labeler)
    fshot = pd.read_csv(data_dir + labeler + 'TCV_' + str(shot) + '_LSTM_det.csv') #+ '/' + machine_id + '_' 
    fshot = fshot.reset_index()
    fshot = fshot.reset_index()    
    # fshot = remove_disruption_points(fshot)
    return fshot, fshot.time.values

#receives numpy array (with two axes) and computes mode along axis 0. If there is no one specific mode (because several values have the same count),
#the mode for that index will be -1. 
def calc_mode(values):
    assert len(values.shape) == 2
    assert values.shape[0] > values.shape[1] #if not, incoming array axes must be swapped!
    if values.shape[1] == 1:
        return np.squeeze(values)
    modes = []
    for v_id, v in enumerate(values):
        # print(v)
        if len(np.unique(v)) == values.shape[1]:
            modes += [-1,]
        else:
            # print(stats.mode(v))
            modes += [stats.mode(v)[0][0]]
    # print(modes)
    modes = np.asarray(modes)
    modes = np.squeeze(modes)
    return modes

def calc_mode_remove_consensus(values):
    assert len(values.shape) == 2
    assert values.shape[0] > values.shape[1] #if not, incoming array axes must be swapped!
    if values.shape[1] == 1:
        return np.squeeze(values)
    modes = []
    for v_id, v in enumerate(values):
        # print(v)
        if len(np.unique(v)) == values.shape[1]:
            modes += [-1,]
        elif len(np.unique(v)) == 1: #consensus, remove
            modes += [-2,]
        else:
            modes += [stats.mode(v)[0][0]]
    # print(modes)
    modes = np.asarray(modes)
    modes = np.squeeze(modes)
    return modes

def calc_consensus(values):
    assert len(values.shape) == 2
    assert values.shape[0] > values.shape[1] #if not, incoming array axes must be swapped!
    if values.shape[1] == 1:
        return np.squeeze(values)
    consensus_labels = []
    # print(values.shape)
    # print('computing modes')
    for v_id, v in enumerate(values):
        vals = np.unique(v)[0]
        if len(np.unique(v)) == 1: #if there is a single label
            consensus_labels += [vals]
        else:
            consensus_labels += [-1,]
            
    consensus_labels = np.asarray(consensus_labels)
    consensus_labels = np.squeeze(consensus_labels)
    return consensus_labels

def calc_trans_mode(values, gaussian_hinterval):
    return calc_elm_mode(values, gaussian_hinterval)

def calc_elm_mode(values, gaussian_hinterval):
    full_interval = gaussian_hinterval*2
    assert len(values.shape) == 2
    assert values.shape[0] > values.shape[1]
    no_labelers = values.shape[1]
    if no_labelers == 1:
        return np.squeeze(values)
    modes = []
    full_windows = len(values)//(full_interval)
    remain_window = len(values)%(full_interval)
    # print(full_interval, len(values), full_windows, remain_window)
    elms = []
    cicle = full_windows + 1
    if remain_window == 0:
        cicle -= 1
    for k in range(cicle):
        if k == full_windows:
            w = values[k*full_interval : k * full_interval + remain_window]
        else:
            w = values[k*full_interval : (k+1)*full_interval]
        # print('w', w.shape)
        # print('w', len(w))
        keys = np.arange(len(w) + 1) #maximum of 20 elms in a window
        votes = {key:[] for key in keys}
        for l in range(no_labelers):
            n_elms = np.sum(w[:, l]).astype(int)
            # print(votes)
            # print(n_elms)
            votes[n_elms] += [l] #labeler l says there's n_elms in this window. 
        # print('votes', votes)
        maj_n_elms = max(len(v) for v in votes.values()) #get the number of labelers who are part of the majority party. (there can also be a tie.)
        # print('maj_n_elms', maj_n_elms)
        max_keys = [key for key, value in votes.items() if len(value) == maj_n_elms] #if more than 1 possibility exists for number of elms in this window:
        # print('max_keys', max_keys)
        if len(max_keys) > 1: #if there is no majority agreement
            elms += list(np.ones(len(w)) * -1)
            continue
        #if majority agrees that there are no elms
        if max_keys[0] == 0: 
            elms += list(np.zeros(len(w)))
            continue
        #else, max_keys contains the ids of the labelers who voted for there being a certain no. of elms in the majority.
        # print(w)
        # whs = np.where(w == 1)
        whs = []
        # print(votes[max_keys[0]])
        for l in range(no_labelers):
            if l not in votes[max_keys[0]]:
                # print(l)
                continue
            whs += [[np.where(w[:,l] == 1)[0][0]]]
        whs = np.asarray(whs)
        # print('whs', whs.shape)# whs.shape should be (20, 3)
        # print(whs)
        assert whs.shape[0] == len(votes[max_keys[0]])
        mean_positions = np.mean(whs,axis=0) # mean_positions.shape should be (3,1)
        # print('mean_positions', mean_positions.shape, mean_positions)
        ids = np.round(mean_positions, 0).astype(int) #ids.shape should be(3,1)
        # print('ids', ids.shape, ids)
        corrected = np.zeros(len(w))
        corrected[ids] = 1
        elms += list(corrected)
    
    elms = np.asarray(elms)
    # print(elms.shape)
    # exit(0)
    return elms    
    
def save_dic(dic, dirname):
    # print(dirname)
    # print(dic)
    with open(dirname + '.pkl', 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

def load_dic(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
#will return the n different labels (generated by n labelers) that exist for a given shot,
#at the same times
def get_different_labels(data_dir, shot, labelers, intersect_times):
    labeler_states = []
    labeler_elms = []
    for k, labeler in enumerate(labelers):
        fshot_labeled = pd.read_csv(data_dir+ labeler +'/TCV_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
        fshot_sliced = fshot_labeled.loc[fshot_labeled['time'].round(5).isin(intersect_times)]
        labeler_states += [fshot_sliced['LHD_label'].values]
        labeler_elms += [fshot_sliced['ELM_label'].values]
    # fshot = fshot.loc[fshot['time'].round(5).isin(intersect_times)]
    labeler_states = np.asarray(labeler_states)
    labeler_elms = np.asarray(labeler_elms)
    return labeler_states, labeler_elms
    
def load_exp_params(train_dir):
    params = {}
    with open(train_dir + '/fixed_exp_params.txt') as fp:
        line = fp.readline()
        cnt = 1
        while line:
            # print("Line {}: {}".format(cnt, line.strip()))
            if line == '\n':
                break
            if line[0] == '#':
                line = fp.readline()
                continue
            # print("Line {}: {}".format(cnt, line.strip()))
            key, val = line.split('=')
            params[key.strip()] = val.strip()
            line = fp.readline()
            cnt += 1
    print(params)
    # exit(0)
    return params
