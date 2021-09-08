import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
from decimal import *
getcontext().prec = 3
import math
from collections import OrderedDict
import scipy


class MyBucket():
    def __init__(self):
        self.queue = []
        
    def prepend(self, state_id, ind, val, csum):
        d_id_c = {'state_id': state_id, 'ind':ind, 'val':val, 'cumsum':csum}
        self.queue.insert(0, d_id_c)
    
    def prepend(self, d_id_c):
        self.queue.insert(0, d_id_c)
        
    def state_ids(self,):
        state_ids = []
        # print(self.queue)
        for item in self.queue:
            state_ids.append(item['state_id'])
        return state_ids
    
    def __getitem__(self, state_id):
        for item in self.queue:
            if item['state_id'] == state_id:
                return item
             
    def popitem(self,):
        return self.queue.pop(-1)
    
    def pop_by_index(self, index):
        return self.queue.pop(index)
    
    def pop_by_state(self, state):
        i = 0
        for i in range(len(self.queue)):
            if self.queue[i]['state_id'] == state:
                return self.pop_by_index(i)
        
    def all_but_last(self,):
        return self.queue[:-1]
    
    def all_but_first(self,):
        return self.queue[1:]
    
    def last(self,):
        return self.queue[-1]

   
def smoothen_states_values_gauss(integer_state_values, times, smooth_window_hsize):
    # print(smooth_window_hsize)
    # exit(0)
    mu=0
    sigma=np.sqrt(10)
    steps = 2*smooth_window_hsize + 1 # +20
    steps = np.arange(steps) - steps//2
    norm_cdf = []
    norm_cdf = .5*(1 + scipy.special.erf((steps-mu)/(sigma*np.sqrt(2))))
    norm_cdf=np.insert(norm_cdf,0,0)
    norm_cdf=np.append(norm_cdf,[1])
    states = ['NoState', 'Low', 'Dither', 'High']
    shot_states = np.empty((len(integer_state_values), len(states))) # 4 possible states (no, low, dither, high)

    current_state = integer_state_values[0] #initial_state = No state, 0
    shot_states = [[0, 0, 0, 0]] 
    shot_states[0][current_state] = 1.
    
    bucket = MyBucket()
    d_id_c = {'state_id': current_state, 'ind':len(norm_cdf)-1, 'cumsum':norm_cdf[-1]}
    bucket.prepend(d_id_c) #value: id, cumsum of ids
    for k in range(1, len(integer_state_values)):
        new_state = integer_state_values[k]
        # print('t', round(times[k], 5), end = '') #'label', 'new_state',  new_state, 
        # shot_state = [norm_cdf[0], norm_cdf[0], norm_cdf[0], norm_cdf[0]]
        
        shot_state = np.copy(shot_states[-1])
        
        if new_state != current_state:
            ind = 0
            if new_state in bucket.state_ids():
                item = bucket.pop_by_state(new_state)
                ind = item['ind'] #- 1
                # print(ind)
            # d_id_c = {'state_id': new_state, 'ind':ind, 'val':vals[ind], 'cumsum':norm_cdf[ind]}
            d_id_c = {'state_id': new_state, 'ind':ind, 'cumsum':norm_cdf[ind]} #'val':vals[ind], 
            bucket.prepend(d_id_c)
            # print(' case 1,2', end = '')
        
        remainder = 0
        for item in bucket.all_but_last():
            state = item['state_id']
            ind = item['ind'] + 1
            item['ind'] = ind
            # print(' ind', item['ind'], end = '')
            update = norm_cdf[ind] - item['cumsum']
            # print(update, end= '')#norm_cdf[ind], item['cumsum'])
            item['cumsum'] = norm_cdf[ind]
            remainder += update
            shot_state[state] = item['cumsum']
            
        if remainder < 0:
            print('something here is wrong. please check')
            exit(0)
        while(remainder>0):
            item = bucket.last()
            # state, csum, val = item['state_id'], item['cumsum'], item['val']
            state, csum = item['state_id'], item['cumsum']
            # print(' remainder', remainder, end= '')
       
            max_removal = csum - norm_cdf[0]#/3 #csum can go down to 0
            temp = remainder - max_removal
            # print('temp', round(temp, 6), 'csum', round(csum,5), round(csum + temp,5), 'max_removal', max_removal, end='')
            # print(' remainder', remainder, end = '')
            if temp > 1e-7: #stuff left to remove. this state can be removed from bucket
                bucket.popitem()
                remainder -= max_removal
                shot_state[state] = shot_states[-1][state] - max_removal
            else:
                item['cumsum'] = csum - remainder
                shot_state[state] = item['cumsum']
                item['ind'] -= 1
                remainder = 0
            
            if item['cumsum'] <= norm_cdf[0]: #/3?
                # print('FINISHED THIS TRANS')
                bucket.pop_by_state(item['state_id'])
                shot_state[state] = 0 #norm_cdf[0]#/3
            # print(' ind', item['ind'], end= '')
        current_state = new_state
        # shot_states.append((shot_states[-1] + cat_to_add))
        shot_states.append(shot_state)
        # print(k, shot_state, round(sum(shot_states[-1]),8))
        assert round(sum(shot_states[-1]),8) == 1.
        
    shot_states = np.asarray(shot_states)
    
    shot_states = np.concatenate((shot_states[len(steps)//2:,:], shot_states[-len(steps)//2+1:,:]), axis=0)
    
    none = shot_states[:,0].round(5)
    low = shot_states[:,1].round(5)
    dither = shot_states[:,2].round(5)
    high = shot_states[:,3].round(5)
    
    shot_states[:,0] = np.zeros(len(shot_states))
    # for k in range(len(shot_states)):
        # assert(sum(shot_states[k]) == 1)
        # if(sum(shot_states[k]) != 1):
        #     print('something wrong here', shot_states[k], sum(shot_states[k]))
        
    none = shot_states[:,0].round(5)
    low = shot_states[:,1].round(5)
    dither = shot_states[:,2].round(5)
    high = shot_states[:,3].round(5)    
    # exit(0)
    
    
    # exit(0)
    return none, low, high, dither#, trans_dic

#Receives: list of binary values denoting ELMs (0 for no, 1 for yes) and the size of half the window over which
#those values will be smoothened, whether for training purposes, or for prediction/evaluation purposes
def smoothen_elm_values(binary_elm_values, smooth_window_hsize):
    padded_binary_elm_values = np.zeros(len(binary_elm_values) + 2*smooth_window_hsize)
    padded_binary_elm_values[smooth_window_hsize : len(binary_elm_values)+smooth_window_hsize] = binary_elm_values
    assert(smooth_window_hsize >= 1)
    smoothed_elms = np.zeros(len(binary_elm_values))
    for k in range(len(binary_elm_values)):
        elms_around = padded_binary_elm_values[k:k+2*smooth_window_hsize+1]
        if not np.array_equal(elms_around, np.zeros(smooth_window_hsize*2+1)):
            one_hot_elms_around = np.argwhere(elms_around != 0)
            distance = np.abs(smooth_window_hsize - one_hot_elms_around)
            smoothed_elms[k] = np.max(1/(np.exp((distance**2)/(2.*np.sqrt(smooth_window_hsize)**2))))
    return(pd.Series(smoothed_elms.round(5)), pd.Series(1-smoothed_elms.round(5)))

#wrapper function for all events, not just elms
def smoothen_event_values(binary_event_values, smooth_window_size):
    return smoothen_elm_values(binary_event_values, smooth_window_size)[0]
