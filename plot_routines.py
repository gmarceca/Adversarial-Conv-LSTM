import matplotlib.pyplot as plt
import numpy as np
from helper_funcs import *
from matplotlib import colors as mcolors
from window_functions import *
# import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

def my_plot_pd_signals_states(scalars, times, states):
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 26}
    import matplotlib
    matplotlib.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
    print(scalars.shape)
    fig = plt.figure(figsize = (19, 6))
    leg = []
    p1 = fig.add_subplot(2,1,1)

    temp = scalars
    temp_pos = temp - np.min(temp)
    temp_norm = temp_pos / np.max(temp_pos)
    scalars = temp_norm

    p1.plot(times,scalars, label='PD')
    p1.grid()
    p1.set_ylabel('PD (norm.)')
    p1.set_xlabel('t(s)')
    p1.xaxis.set_ticklabels([])
    # positions = np.where(elms[:,0] > .9)[0]
    # l = 'ELM'
    # for pos in positions:
    #     p4.axvline(x = times[pos], linestyle='-', color='b', alpha=1., label = l, linewidth=2)
    #     l = "_nolegend_"
    # 

    p2 = fig.add_subplot(2,1,2)
    cs = ['g',colors['gold'],'r']
    for s_ind, s in enumerate(['Low', 'Dither', 'High']):
        print(s)
        #p2.plot(times,states[:, s_ind], color=cs[s_ind], label=s)
        # print(states.shape, s_ind, type(states[:, s_ind] >= .5), (states[:, s_ind] >= .5).shape, min_d, max_d, np.squeeze(times).shape)
        p2.fill_between(np.squeeze(times), 0, 1, where=list(states[:, s_ind] >= .33), facecolor=cs[s_ind], alpha=0.1, label=s)

    p2.legend(loc=2, prop={'size': 22})
    p2.set_xlabel('t(s)')
    p2.set_ylabel('Prob.')
    p2.grid()
    # plt.tight_layout()
    plt.show()

def plot_pd_signals_states(elms, scalars, times, states):
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 26}
    import matplotlib
    matplotlib.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
    print(scalars.shape)
    fig = plt.figure(figsize = (19, 6))
    temp = np.asarray(elms[:,0])
    elms[:,0] = (temp==1.).astype(float)
    leg = []
    p1 = fig.add_subplot(2,1,1)
    
    temp = scalars
    temp_pos = temp - np.min(temp)
    temp_norm = temp_pos / np.max(temp_pos)
    scalars = temp_norm
    
    p1.plot(times,scalars, label='PD')
    p1.grid()
    p1.set_ylabel('PD (norm.)')
    p1.set_xlabel('t(s)')
    p1.xaxis.set_ticklabels([])
    # positions = np.where(elms[:,0] > .9)[0]
    # l = 'ELM'
    # for pos in positions:
    #     p4.axvline(x = times[pos], linestyle='-', color='b', alpha=1., label = l, linewidth=2)
    #     l = "_nolegend_"
    # 
    
    p2 = fig.add_subplot(2,1,2)
    cs = ['g',colors['gold'],'r']
    for s_ind, s in enumerate(['Low', 'Dither', 'High']):
        p2.plot(times,states[:, s_ind], color=cs[s_ind], label=s)
        # print(states.shape, s_ind, type(states[:, s_ind] >= .5), (states[:, s_ind] >= .5).shape, min_d, max_d, np.squeeze(times).shape)
        # p4.fill_between(np.squeeze(times), 0, 1, where=list(states[:, s_ind] >= .5), facecolor=cs[s_ind], alpha=0.1, label=s)

            
    p2.legend(loc=2, prop={'size': 22})
    p2.set_xlabel('t(s)')
    p2.set_ylabel('Prob.')
    p2.grid()
    # plt.tight_layout()
    plt.show()

def plot_all_signals_all_trans(elms, scalars, times, trans, machine_id): 
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 26}
    import matplotlib
    matplotlib.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
    print(scalars.shape)
    fig = plt.figure(figsize = (19, 5))
    temp = np.asarray(elms[:,0])
    elms[:,0] = (temp==1.).astype(float)
    leg = []
    p4 = fig.add_subplot(1,1,1)
    if machine_id == 'TCV':
        p4.plot(times,scalars[:,1], label='PD')
        p4.plot(times,scalars[:,0], label='FIR')
    elif machine_id == 'JET':
        p4.plot(times,scalars[:,0], label='FIR')
        p4.plot(times,scalars[:,1], label='PD')
    p4.grid()
    p4.set_ylabel('Signal values (norm.)')
    p4.set_xlabel('t(s)')
    positions = np.where(elms[:,0] > .9)[0]
    l = 'ELM'
    # for pos in positions:
    #     p4.axvline(x = times[pos], linestyle='-', color='b', alpha=1., label = l, linewidth=2)
    #     l = "_nolegend_"
    
    
    temp = np.asarray(trans[:,:])
    trans[:,:] = (temp==1.).astype(float)
    temp2 = get_trans_ids()
    
    for i in range(6):
        positions = np.where(trans[:,i] == 1)[0]
        l = temp2[i]
        if len(positions) == 0:
            continue
        # leg += [temp2[i]]
        for pos in positions:
            p4.axvline(x = times[pos], linestyle='-', color=cs[i], alpha=1., label=l, linewidth=2)
            l = "_nolegend_"
            
    p4.legend(loc=2, prop={'size': 22}, ncol=2)
    plt.tight_layout()
    plt.show()
    
    
def plot_pd_states_trans_elms(elms, pd, times, trans, states):
    # print(matplotlib.text.Text['fontproperties'])
    # exit(0)
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 26}
    
    plt.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
    print(pd.shape)
    fig = plt.figure(figsize = (19, 8)) #figsize = (19, 5)

    p1 = fig.add_subplot(4,1,1)
    p1.plot(times,pd, label='PD')
    p1.grid()
    p1.set_ylabel('PD (norm.)')
    p1.xaxis.set_ticklabels([])
    p1.legend(loc=2, prop={'size': 22})
    
    
    p2 = fig.add_subplot(4,1,2)
    for t_ind, t in enumerate(get_trans_ids()):
        p2.plot(times,trans[:,t_ind], label=t, color=cs[t_ind])
    p2.set_ylim([-.1,1.1])
    p2.grid()
    p2.set_ylabel('Prob.')
    p2.xaxis.set_ticklabels([])
    p2.legend(loc=2, prop={'size': 22}, ncol=3)
    
    
    cs = ['g',colors['gold'],'r']
    p3 = fig.add_subplot(4,1,3)
    for s_ind, s in enumerate(['Low', 'Dither', 'High']):
        p3.plot(times,states[:, s_ind], color=cs[s_ind], label=s)
    p3.set_ylim([-.1,1.1])
    p3.grid()
    p3.set_ylabel('Prob.')
    p3.xaxis.set_ticklabels([])
    p3.legend(loc=2, prop={'size': 22})
    
    
    
    
    
    p4 = fig.add_subplot(4,1,4)
    p4.plot(times,elms[:,0], label='ELM', color='b')
    p4.set_ylim([-.1,1.1])
    p4.grid()
    p4.set_ylabel('Prob.')
    p4.set_xlabel('t(s)')
    
    
    p4.legend(loc=2, prop={'size': 22}, ncol=2)
    

    # plt.tight_layout(rect=[0.02, 0, 1, 0.92])
    plt.show()
    
    
def plot_pd_elms(elms, pd, times):
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 30}
    
    plt.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
    print(pd.shape)
    fig = plt.figure(figsize = (19, 8)) #figsize = (19, 5)

    p1 = fig.add_subplot(2,1,1)
    p1.plot(times,pd, label='PD')
    p1.grid()
    p1.set_ylabel('PD (norm.)')
    p1.xaxis.set_ticklabels([])
    p1.legend(loc=1, prop={'size': 24})
    
    
    p2 = fig.add_subplot(2,1,2)
    p2.plot(times,elms[:,0], label='ELM', color='b')
    p2.set_ylim([-.1,1.1])
    p2.grid()
    p2.set_ylabel('Prob.')
    p2.set_xlabel('t (s)')
    # p2.xaxis.set_ticklabels([])
    p2.legend(loc=1, prop={'size': 24})
    

    # plt.tight_layout(rect=[0.02, 0, 1, 0.92])
    # plt.tight_layout()
    plt.show()
 
 
def plot_all_signals_states(elms, scalars, times, states):
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 26}
    import matplotlib
    matplotlib.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
    print(scalars.shape)
    fig = plt.figure(figsize = (19, 6))
    temp = np.asarray(elms[:,0])
    elms[:,0] = (temp==1.).astype(float)
    leg = []
    p4 = fig.add_subplot(1,1,1)
    
    for k in range(4):
        temp = scalars[:,k]
        temp_pos = temp - np.min(temp)
        temp_norm = temp_pos / np.max(temp_pos)
        scalars[:,k] = temp_norm
    
    p4.plot(times,scalars[:,2]-np.min(scalars[:,2]), label='PD')
    p4.plot(times,scalars[:,0], label='FIR')
    p4.plot(times,scalars[:,1], label='DML')
    p4.plot(times,scalars[:,3], label='IP')
    p4.grid()
    p4.set_ylabel('Signal values (norm.)')
    p4.set_xlabel('t(s)')
    positions = np.where(elms[:,0] > .9)[0]
    l = 'ELM'
    # for pos in positions:
    #     p4.axvline(x = times[pos], linestyle='-', color='b', alpha=1., label = l, linewidth=2)
    #     l = "_nolegend_"
    # 
    
    cs = ['g',colors['gold'],'r']
    for s_ind, s in enumerate(['Low', 'Dither', 'High']):
        # p4.plot(times,states[:, s_ind], color=cs[s_ind], label=s)
        # print(states.shape, s_ind, type(states[:, s_ind] >= .5), (states[:, s_ind] >= .5).shape, min_d, max_d, np.squeeze(times).shape)
        p4.fill_between(np.squeeze(times), 0, 1, where=list(states[:, s_ind] >= .5), facecolor=cs[s_ind], alpha=0.1, label=s)

            
    p4.legend(loc=2, prop={'size': 22})
    plt.tight_layout()
    plt.show()
    
def plot_pd_trans(elms, pd, times, trans):
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 30}
    
    plt.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
    print(pd.shape)
    fig = plt.figure(figsize = (19, 8)) #figsize = (19, 5)

    p1 = fig.add_subplot(2,1,1)
    p1.plot(times,pd, label='PD')
    p1.grid()
    p1.set_ylabel('PD (norm.)')
    p1.xaxis.set_ticklabels([])
    p1.legend(loc=1, prop={'size': 24})
    
    
    p2 = fig.add_subplot(2,1,2)
    for t_ind, t in enumerate(get_trans_ids()):
        p2.plot(times,trans[:,t_ind], label=t, color=cs[t_ind])
    p2.set_ylim([-.1,1.1])
    p2.grid()
    p2.set_ylabel('Prob.')
    p2.set_xlabel('t(s)')
    p2.legend(loc=2, prop={'size': 22}, ncol=2)
    


    # plt.tight_layout(rect=[0.02, 0, 1, 0.92])
    # plt.tight_layout()
    plt.show()

def plot_windows_blocks(conv_windows, transition_blocks, shifted_transition_blocks, times, shot, stride=10, conv_size=40, block_size=10, look_ahead=10):
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}
    
    plt.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['r','g','y',colors['teal'],'c','m',colors['orange']]
    fig = plt.figure(figsize = (19, 8)) #figsize = (19, 5)

    p1 = fig.add_subplot(3,1,1)
    p1.set_ylabel('signal values')
    # p1.plot(times,pd, label='PD')
    clist =['#99ff99', '#ffff99', '#ffcc99', '#99ccff']
    for w in range(len(conv_windows)):
    # for w in range(3,4):
        # print('w', w)
        times_to_plot = times[w*stride : w*stride+conv_size]
        # locations = np.arange(0,40)
        # print(times_to_plot)
        # ax = plt.axes()
        
        p1.plot(times_to_plot, conv_windows[w, :, 0], color='m', label='FIR')
        p1.plot(times_to_plot, conv_windows[w, :, 1], color='g', label='DML')
        p1.plot(times_to_plot, conv_windows[w, :, 2], color='b', label='PD')
        p1.plot(times_to_plot, conv_windows[w, :, 3], color='c', label='IP')
    p1.set_xlim(times[0], times[-1])    
    p1.set_title('Encoder Input (windowed signal values)' + str(conv_windows.shape))
    p1.grid()
    # p1.set_ylabel('PD (norm.)')
    # p1.xaxis.set_ticklabels(times_to_plot)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # p1.legend(loc=2, prop={'size': 22})
    plt.legend(by_label.values(), by_label.keys())
    
    p2 = fig.add_subplot(3,1,2)
    p2.set_title('Decoder Input (sequence of blocks)' + str(transition_blocks.shape) + '. Block size =' + str(block_size))
    p2.set_ylabel('P(transition)')
    # for w in range(len(conv_windows)):
    # print(transition_blocks.shape)
    labels = get_trans_ids()+['no_trans']
    p2.set_xlim([times[0], times[-1]])
    for k in range(len(transition_blocks)):
        times_to_plot_block = times[k*block_size + look_ahead: k*block_size+block_size+look_ahead+1]
        # print(times_to_plot_block)
        block = transition_blocks[k]
        to_color = np.where(block==1)[0]#[0]
        # print(to_color, len(to_color))
        if len(to_color) > 0:
            c_ind = int(to_color[0])
            p2.axvspan(times_to_plot_block[0], times_to_plot_block[-1], color=cs[c_ind], alpha=.3, label=labels[c_ind])
        else:
            p2.axvspan(times_to_plot_block[0], times_to_plot_block[-1], color='blue', alpha=.3, label='START')
    right_look_ahead_id = k*block_size+block_size+look_ahead
    times_to_plot = times[right_look_ahead_id: right_look_ahead_id + look_ahead + 1]
    p2.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.3, label='Look-ahead')
    right_look_ahead_id = right_look_ahead_id + look_ahead
    times_to_plot = times[right_look_ahead_id: ]
    if len(times_to_plot) > 0:
        p2.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.7, label='Remainder')
    times_to_plot = times[0: look_ahead+1]
    p2.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.3, label='Look-ahead')
    
    # p2.set_xticklabels(times_to_plot)
    # exit(0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), ncol=len(labels))
   
    p3 = fig.add_subplot(3,1,3)
    p3.set_title('Decoder Output (shifted blocks)' + str(shifted_transition_blocks.shape) + '. Block size =' + str(block_size))
    # for w in range(len(conv_windows)):
    # print(transition_blocks.shape)
    labels = get_trans_ids()+['no_trans']
    p3.set_xlim([times[0], times[-1]])
    for k in range(len(transition_blocks)):
        times_to_plot = times[k*block_size +look_ahead: k*block_size+block_size+look_ahead+1]
        # print(times_to_plot)
        block = shifted_transition_blocks[k]
        to_color = np.where(block==1)[0][0]
        # print(to_color)
        p3.axvspan(times_to_plot[0], times_to_plot[-1], color=cs[to_color], alpha=.3, label=labels[to_color])
    right_look_ahead_id = k*block_size+block_size+look_ahead
    times_to_plot = times[right_look_ahead_id: right_look_ahead_id + look_ahead + 1]
    p3.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.3, label='Look-ahead')
    right_look_ahead_id = right_look_ahead_id + look_ahead
    times_to_plot = times[right_look_ahead_id: ]
    if len(times_to_plot) > 0:
        p3.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.7, label='Remainder')
    times_to_plot = times[0: look_ahead+1]
    p3.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.3, label='Look-ahead')
    
    p3.set_xlabel('t(s)')
    p3.set_ylabel('P(transition)')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), ncol=len(labels))
    plt.suptitle('#' + shot)
    plt.tight_layout()
    plt.show()
    
def plot_windows_blocks_states(conv_windows, transition_blocks, shifted_transition_blocks, times, shot, stride=10, conv_size=40, block_size=10, look_ahead=10):
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}
    
    plt.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['g','y','r']
    fig = plt.figure(figsize = (19, 8)) #figsize = (19, 5)

    p1 = fig.add_subplot(3,1,1)
    p1.set_ylabel('signal values')
    # p1.plot(times,pd, label='PD')
    clist =['#99ff99', '#ffff99', '#ffcc99', '#99ccff']
    for w in range(len(conv_windows)):
    # for w in range(3,4):
        # print('w', w)
        times_to_plot = times[w*stride : w*stride+conv_size]
        
        p1.plot(times_to_plot, conv_windows[w, :, 0], color='m', label='FIR')
        p1.plot(times_to_plot, conv_windows[w, :, 1], color='g', label='DML')
        p1.plot(times_to_plot, conv_windows[w, :, 2], color='b', label='PD')
        p1.plot(times_to_plot, conv_windows[w, :, 3], color='c', label='IP')
    p1.set_xlim(times[0], times[-1])    
    p1.set_title('Encoder Input (windowed signal values)' + str(conv_windows.shape))
    p1.grid()
    # p1.set_ylabel('PD (norm.)')
    # p1.xaxis.set_ticklabels(times_to_plot)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # p1.legend(loc=2, prop={'size': 22})
    plt.legend(by_label.values(), by_label.keys())
    
    p2 = fig.add_subplot(3,1,2)
    p2.set_title('Decoder Input (sequence of blocks)' + str(transition_blocks.shape) + '. Block size =' + str(block_size))
    p2.set_ylabel('P(transition)')
    # for w in range(len(conv_windows)):
    # print(transition_blocks.shape)
    # labels = get_trans_ids()+['no_trans']
    labels = ['L', 'D', 'H']
    p2.set_xlim([times[0], times[-1]])
    for k in range(len(transition_blocks)):
        times_to_plot_block = times[k*block_size + look_ahead: k*block_size+block_size+look_ahead+1]
        # print(times_to_plot_block)
        block = transition_blocks[k]
        to_color = np.where(block==1)[0]#[0]
        # print(to_color, len(to_color))
        if len(to_color) > 0:
            c_ind = int(to_color[0])
            p2.axvspan(times_to_plot_block[0], times_to_plot_block[-1], color=cs[c_ind], alpha=.3, label=labels[c_ind])
        else:
            p2.axvspan(times_to_plot_block[0], times_to_plot_block[-1], color='blue', alpha=.3, label='START')
    right_look_ahead_id = k*block_size+block_size+look_ahead
    times_to_plot = times[right_look_ahead_id: right_look_ahead_id + look_ahead + 1]
    p2.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.3, label='Look-ahead')
    right_look_ahead_id = right_look_ahead_id + look_ahead
    times_to_plot = times[right_look_ahead_id: ]
    if len(times_to_plot) > 0:
        p2.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.7, label='Remainder')
    times_to_plot = times[0: look_ahead+1]
    p2.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.3, label='Look-ahead')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), ncol=len(labels))
   
    p3 = fig.add_subplot(3,1,3)
    p3.set_title('Decoder Output (shifted blocks)' + str(shifted_transition_blocks.shape) + '. Block size =' + str(block_size))

    labels = ['L', 'D', 'H']
    p3.set_xlim([times[0], times[-1]])
    for k in range(len(transition_blocks)):
        times_to_plot = times[k*block_size +look_ahead: k*block_size+block_size+look_ahead+1]
        # print(times_to_plot)
        block = shifted_transition_blocks[k]
        to_color = np.where(block==1)[0][0]
        # print(to_color)
        p3.axvspan(times_to_plot[0], times_to_plot[-1], color=cs[to_color], alpha=.3, label=labels[to_color])
    right_look_ahead_id = k*block_size+block_size+look_ahead
    times_to_plot = times[right_look_ahead_id: right_look_ahead_id + look_ahead + 1]
    p3.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.3, label='Look-ahead')
    right_look_ahead_id = right_look_ahead_id + look_ahead
    times_to_plot = times[right_look_ahead_id: ]
    if len(times_to_plot) > 0:
        p3.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.7, label='Remainder')
    times_to_plot = times[0: look_ahead+1]
    p3.axvspan(times_to_plot[0], times_to_plot[-1], color=colors['black'], alpha=.3, label='Look-ahead')
    
    p3.set_xlabel('t(s)')
    p3.set_ylabel('P(state)')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), ncol=len(labels))
    plt.suptitle('#' + shot)
    plt.tight_layout()
    plt.show()
    
def plot_windows_prediction(shot_signals, decoded_sequence, times, shot, convolutional_stride, conv_w_size, block_size, fname):
    font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16}
    
    plt.rc('font', **font)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
    fig = plt.figure(figsize = (19, 8)) #figsize = (19, 5)

    p1 = fig.add_subplot(2,1,1)
    p1.set_ylabel('signal values')
    # p1.plot(times,pd, label='PD')
    clist =['#99ff99', '#ffff99', '#ffcc99', '#99ccff']
    # for w in range(len(shot_signals)):
    # # for w in range(3,4):
    #     print(shot_signals.shape)
    #     print('w', w)
    times_to_plot = times

    p1.plot(times_to_plot, shot_signals[:, 0], color='m', label='FIR')
    p1.plot(times_to_plot, shot_signals[:, 1], color='g', label='DML')
    p1.plot(times_to_plot, shot_signals[:, 2], color='b', label='PD')
    p1.plot(times_to_plot, shot_signals[:, 3], color='c', label='IP')
        # frac = (w%4)/4
        # p1.axvspan(times_to_plot[0], times_to_plot[-1], facecolor=clist[w%4], alpha=.7, ymin = frac, ymax = frac+.25)
    p1.set_title('Encoder Input (windowed signal values)' + str(shot_signals.shape))
    p1.grid()
    # p1.set_ylabel('PD (norm.)')
    # p1.xaxis.set_ticklabels(times_to_plot)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # p1.legend(loc=2, prop={'size': 22})
    plt.legend(by_label.values(), by_label.keys(), loc=2)
    
    p2 = fig.add_subplot(2,1,2)
    p2.set_title('Decoder Output (sequence of transition blocks)' + str(decoded_sequence.shape))
    p2.set_ylabel('P(transition)')
    # for w in range(len(conv_windows)):
    # print(transition_blocks.shape)
    labels = get_trans_ids()+['no_trans']
    # print(decoded_sequence[:1000])
    for k in range(len(decoded_sequence)):
        times_to_plot = times[k*block_size : k*block_size+block_size]
        # print(times_to_plot)
        block = decoded_sequence[k, 0]
        # block = 6
        # print(block.shape, block)
        # to_color = np.where(block==1)[0]#[0]
        # print(to_color, len(to_color))
        # if len(to_color) > 0:
            # c_ind = int(to_color[0])
        if block != 6:
            p2.axvspan(times_to_plot[0], times_to_plot[-1], facecolor=cs[block], alpha=.3, label=labels[block])
        else:
            p2.axvspan(times_to_plot[0], times_to_plot[-1], facecolor='white', alpha=.3, label='No trans')
        # else:
    # exit(0)
    p2.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # print(decoded_sequence.shape)
    # print(decoded_sequence[:20])
   
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname)
    
    
# def plot_attention_prediction(shot_signals, decoded_sequence, trans_detected, attention_weights_sequence, times, shot, convolutional_stride,
#                               conv_w_size, block_size, fname, max_source_sentence_chars,
#                               max_train_target_words, look_ahead, num_channels):
#     font = {'family' : 'normal',
#         # 'weight' : 'bold',
#         'size'   : 16}
#     blocks = trans_detected
#     # blocks = np.arange(11)
#     # blocks_in_subseq = timesteps // block_size
#     # encoder_inputs_to_blocks = np.empty(len(blocks), timesteps, num_channels)
#     blocks_per_source_sentence = max_train_target_words
#     target_chars_per_sentence = blocks_per_source_sentence * block_size
#     remainder = max_source_sentence_chars - target_chars_per_sentence- 2*look_ahead
#     
#     handler = PdfPages(fname)
#     # print(blocks_per_source_sentence, max_source_sentence_chars)
#     # print(attention_weights_sequence.shape)
#     # exit(0)
#     cumul=0
#     for ind, block_ind in enumerate(blocks):
#         # signal_start_ind = block_ind * block_size - remainder
#         # signal_end_ind = signal_start_ind + max_source_sentence_chars
#         subseqs_until_block = block_ind // blocks_per_source_sentence
#         signal_start_ind = (subseqs_until_block) * max_source_sentence_chars -2*look_ahead*subseqs_until_block - remainder*subseqs_until_block
#         signal_end_ind = signal_start_ind + max_source_sentence_chars
#         # print(block_ind, block_ind // blocks_per_source_sentence, signal_start_ind, signal_end_ind, len(shot_signals))
#         if signal_end_ind > len(shot_signals):
#             print('warning, you are trying to plot a block outside the range specified in the call to the main function.')
#             handler.close()
#             sys.stdout.flush()
#             return
#             # exit(0)
#         shot_signals_sliced = shot_signals[signal_start_ind:signal_end_ind]
#         times_sliced = times[signal_start_ind:signal_end_ind]
#         
#         # block_start_ind = block_ind
#         # block_end_ind = block_start_ind + blocks_per_source_sentence
#         # decoded_sequence_sliced = decoded_sequence[block_start_ind : block_end_ind]
#         # attention_weights_seq_sliced = attention_weights_sequence[block_start_ind : block_end_ind, :, 0]
#         
#         block_start_ind = subseqs_until_block * blocks_per_source_sentence
#         block_end_ind = block_start_ind + blocks_per_source_sentence
#         decoded_sequence_sliced = decoded_sequence[block_start_ind : block_end_ind]
#         attention_weights_seq_sliced = attention_weights_sequence[block_start_ind : block_end_ind, :, 0]
#         
#         plt.rc('font', **font)
#         colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#         cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
#         cs = ['g',colors['gold'],'r']
#         fig = plt.figure(figsize = (19, 8)) #figsize = (19, 5)
#     
#         p1 = fig.add_subplot(2,1,1)
#         p1.set_ylabel('signal values')
#         # p1.plot(times,pd, label='PD')
#         clist =['#99ff99', '#ffff99', '#ffcc99', '#99ccff']
#         
#     
#         p1.plot(times_sliced, shot_signals_sliced[:, 0], color='m', label='FIR')
#         p1.plot(times_sliced, shot_signals_sliced[:, 1], color='g', label='DML')
#         p1.plot(times_sliced, shot_signals_sliced[:, 2], color='b', label='PD')
#         p1.plot(times_sliced, shot_signals_sliced[:, 3], color='c', label='IP')
#             # frac = (w%4)/4
#             # p1.axvspan(times_to_plot[0], times_to_plot[-1], facecolor=clist[w%4], alpha=.7, ymin = frac, ymax = frac+.25)
#         p1.set_title('Encoder Input (windowed signal values)' + str(shot_signals_sliced.shape))
#         p1.grid(zorder=0)
#         p1.set_xlim([times_sliced[0], times_sliced[-1]])
#         # p1.set_ylabel('PD (norm.)')
#         # p1.xaxis.set_ticklabels(times_to_plot)
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         # p1.legend(loc=2, prop={'size': 22})
#         plt.legend(by_label.values(), by_label.keys(), loc=2)
#         
#         p2 = fig.add_subplot(2,1,2)
#         p2.set_title('Decoder Output (sequence of transition blocks)' + str(decoded_sequence_sliced.shape))
#         p2.set_ylabel('P(transition)')
#         p2.set_xlim([times_sliced[0], times_sliced[-1]])
#         labels = get_trans_ids()+['no_trans']
#         labels = ['L', 'D', 'H']
#         
#         times_sliced_look_ahead = times_sliced[look_ahead : - look_ahead - remainder]
#         # print(times_sliced_look_ahead[0], times_sliced_look_ahead[-1])
#         
#         attn_weights = convolve_att_weights(conv_w_size, convolutional_stride, attention_weights_seq_sliced)
#         # attn_weights = attention_weights_seq_sliced
#         # print(attn_weights_convolved.shape)
#         # exit(0)
#         # ylims = p1.get_ylim()
#         for k in range(len(decoded_sequence_sliced)):
#             times_block_sliced = times_sliced_look_ahead[k*block_size : k*block_size+block_size]
#             # print(k, times_block_sliced[0], times_block_sliced[-1])
#             # times_block_sliced = times_sliced[k*block_size : k*block_size+block_size]
#             # print(times_to_plot)
#             block = decoded_sequence_sliced[k] - 1
#             # block = 6
#             # print(block.shape, block)
#             # to_color = np.where(block==1)[0]#[0]
#             # print(to_color, len(to_color))
#             # if len(to_color) > 0:
#                 # c_ind = int(to_color[0])
#             
#             # if(block_ind % len(decoded_sequence_sliced)) == k:
#             # highest_att_ids = np.argsort(attention_weights_seq_sliced[k])[-5:]
#             # print(highest_att_ids)
#             if k == block_ind % blocks_per_source_sentence:
#                 # print(k, block_ind, block_size, len(decoded_sequence_sliced))
#                 alpha = 1
#                 # positions = np.arange(19)
#                 # positions *= 10
#                 # positions += 10
#                 # print(attention_weights_seq_sliced.shape)
#                 # p1.set_xticks(positions)
#                 # p1.set_xticklabels(np.round(attention_weights_seq_sliced[:,0], 3))
#                 # print(np.argsort(attention_weights_seq_sliced, axis=0)) #3-highest probability
#                 # print(attention_weights_seq_sliced[k].shape)
#                 sorted_att_ids = np.argsort(attn_weights[k], axis=0) #lowest to highest
#                 # highest_att_ids = np.asarray([0,1])
#                 # print(highest_att_ids)
#                 max_alpha = .5
#                 alphas = np.arange(0, max_alpha, max_alpha/attn_weights.shape[1])[::-1]
#                 alphas = attn_weights[k]
#                 alphas -= np.min(alphas)
#                 distortion_factor = max_alpha / np.max(alphas)
#                 alphas *= distortion_factor
#                 # plt.plot(alphas)
#                 alphas = np.clip(alphas, a_min=0, a_max=max_alpha)
#                 alphas = max_alpha - alphas
#                 # print(alphas)
#                 for att_id_ind, att_id in enumerate(sorted_att_ids):
#                     att_time_st = att_id * convolutional_stride
#                     att_time_end = att_time_st + convolutional_stride 
#                     # att_time_end = att_time_st + conv_w_size - 1
#                     #careful with rounding errors in next line
#                     if times_sliced[att_time_st] == times_sliced[-convolutional_stride]:
#                         att_time_end -= 1
#                     left_limit, right_limit = times_sliced[att_time_st], times_sliced[att_time_end]    
#                     # time_int = times_sliced[att_time_st : att_time_end]
#                     p1.axvspan(left_limit, right_limit, facecolor='black', alpha=alphas[att_id], label=att_id_ind + 1)
#                     # p1.fill_between(times_sliced, ylims[0], ylims[1], where=times_sliced, facecolor='black', alpha=alphas[att_id_ind])
#                     
#                     # p1.text(times_sliced[att_time_st+conv_w_size//2-1], 1.5, np.round(attention_weights_seq_sliced[att_id, 0],3)[0], rotation=90) 
#             else:
#                 alpha = .3
#                 
#             # if block != 6:
#             #     p2.axvspan(times_block_sliced[0], times_block_sliced[-1], facecolor=cs[block], alpha=alpha, label=labels[block])
#             # else:
#             #     p2.axvspan(times_block_sliced[0], times_block_sliced[-1], facecolor='gray', alpha=alpha, label='No trans')
#             # print(block)
#             # print(times_block_sliced[0])
#             # print(times_block_sliced[-1])
#             p2.axvspan(times_block_sliced[0], times_block_sliced[-1], facecolor=cs[block], alpha=alpha, label=labels[block])
#         # exit(0)
#         p2.grid()
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         plt.legend(by_label.values(), by_label.keys())
#         
#        
#         plt.tight_layout()
#         # plt.show()
#         # plt.savefig(fname)
#         fig.savefig(handler, format='pdf')
#         plt.close('all')
#     handler.close()
#     sys.stdout.flush()
# 
# def convolve_att_weights(conv_w_size, convolutional_stride, attention_weights_seq_sliced):
#     attn_weights_convolved = []
#     windows_per_conv = conv_w_size//convolutional_stride
#     num_windows = attention_weights_seq_sliced.shape[1]
#     for i in range(num_windows + windows_per_conv - 1):
#         inds = np.arange(i - windows_per_conv +1, i + 1)
#         inds = inds[np.logical_and(inds>= 0, inds <= num_windows - 1)]
#         vals = attention_weights_seq_sliced[:, inds]
#         attn_weights_convolved.append(np.sum(vals, axis=1)/vals.shape[1])
#     attn_weights_convolved = np.asarray(attn_weights_convolved).swapaxes(0,1)
#     return attn_weights_convolved
# 
# def plot_attention_matrix(shot_signals, decoded_sequence, trans_detected, attention_weights_sequence, times, shot, convolutional_stride,
#                           conv_w_size, block_size, fname, max_source_sentence_chars,
#                           max_train_target_words, look_ahead, num_channels):
#     font = {'family' : 'normal',
#         # 'weight' : 'bold',
#         'size'   : 16}
#     # blocks = trans_detected
#     # blocks = np.arange(22)  
#     # blocks_per_source_sentence = (max_source_sentence_chars - 2*look_ahead) // block_size
#     # assert blocks_per_source_sentence == target_blocks_per_source_sentence
#     blocks_per_source_sentence = max_train_target_words
#     target_chars_per_sentence = blocks_per_source_sentence * block_size
#     remainder = max_source_sentence_chars - target_chars_per_sentence- 2*look_ahead
#     
#     handler = PdfPages(fname)
#     blocks = np.arange(len(decoded_sequence))
#     
#     # assert shot_signals.shape[0] >= decoded_sequence.shape[0]*block_size  + 2*look_ahead
#     # print('in plot_attention_matrix')
#     # print(blocks.shape)
#     # exit(0)
#     for ind, block_ind in enumerate(blocks[::blocks_per_source_sentence]):
#         # print(ind)
#         # signal_start_ind = (block_ind // blocks_per_source_sentence) * max_source_sentence_chars - cumul
#         signal_start_ind = (block_ind // blocks_per_source_sentence) * max_source_sentence_chars -2*look_ahead*ind - remainder*ind
#         signal_end_ind = signal_start_ind + max_source_sentence_chars
#         
#         # print(signal_start_ind, signal_end_ind)
#         if signal_end_ind > len(shot_signals):
#             print('warning, you are trying to plot a block outside the range specified in the call to the main function.')
#             handler.close()
#             sys.stdout.flush()
#             return
#             # exit(0)
#         shot_signals_sliced = shot_signals[signal_start_ind:signal_end_ind]
#         times_sliced = times[signal_start_ind:signal_end_ind]
#         # print(times[signal_start_ind:signal_end_ind])
#         # exit(0)
#         block_start_ind = (block_ind // blocks_per_source_sentence) * blocks_per_source_sentence
#         block_end_ind = block_start_ind + blocks_per_source_sentence
#         decoded_sequence_sliced = decoded_sequence[block_start_ind : block_end_ind]
#         attention_weights_seq_sliced = attention_weights_sequence[block_start_ind : block_end_ind, :, 0]
#         # print(attention_weights_sequence.shape, attention_weights_seq_sliced.shape, decoded_sequence_sliced.shape)
#         # exit(0)
#         plt.rc('font', **font)
#         colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
#         cs = ['r','g','y',colors['teal'],'c','m',colors['black']]
#         cs = ['g','y','r']
#         fig = plt.figure(figsize = (17, 8)) #figsize = (19, 5)
#         # fig = plt.figure()
#         gs1 = GridSpec(12, 12)
#         ax1 = fig.add_subplot(gs1[0:3, 3:])
#         ax2 = fig.add_subplot(gs1[3:,3:])
#         ax3 = fig.add_subplot(gs1[3:, 0:3])
#         
#         # p1 = fig.add_subplot(2,1,1)
#         ax1.set_ylabel('Signal\n values')
#         # p1.plot(times,pd, label='PD')
#         clist =['#99ff99', '#ffff99', '#ffcc99', '#99ccff']
#         
#         # print(times_sliced, times_sliced.shape, shot_signals_sliced.shape, shot_signals.shape, times.shape)
#     
#         ax1.plot(times_sliced, shot_signals_sliced[:, 0], color='m', label='FIR')
#         ax1.plot(times_sliced, shot_signals_sliced[:, 1], color='g', label='DML')
#         ax1.plot(times_sliced, shot_signals_sliced[:, 2], color='b', label='PD')
#         ax1.plot(times_sliced, shot_signals_sliced[:, 3], color='c', label='IP')
#         ax1.xaxis.set_ticks_position('top')
#         # ax1.set_xticklabels([])
#         # plt.show()
#             # frac = (w%4)/4
#             # p1.axvspan(times_to_plot[0], times_to_plot[-1], facecolor=clist[w%4], alpha=.7, ymin = frac, ymax = frac+.25)
#         ax1.set_title('Encoder Input (signal values)' + str(shot_signals_sliced.shape) + '. Window size:' + str(conv_w_size) + '. Stride:' + str(convolutional_stride))
#         ax1.set_xlim(times_sliced[0], times_sliced[-1])
#         ax1.grid()
#         # plt.show()
#         # p1.set_ylabel('PD (norm.)')
#         # p1.xaxis.set_ticklabels(times_to_plot)
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         # p1.legend(loc=2, prop={'size': 22})
#         plt.legend(by_label.values(), by_label.keys(), loc=2)
#         
#         # p2 = fig.add_subplot(2,1,2)
#         
#         # ax3.set_title('Decoder Output (sequence of transition blocks)' + str(decoded_sequence.shape))
#         
#         ax3.set_xlabel('P(transition)')
#         # ax3.set_ylabel('Time(s)')
#         ax3.set_ylabel('Decoder Output (sequence of\n transition blocks)' + str(decoded_sequence.shape))
#         labels = ['L', 'D', 'H']
#         times_sliced_look_ahead = times_sliced[look_ahead : - look_ahead - remainder]
#         
#         for k in range(len(decoded_sequence_sliced)):
#             times_block_sliced = times_sliced_look_ahead[k*block_size : k*block_size+block_size]
#             # block = decoded_sequence_sliced[k, 0]
#             block = decoded_sequence_sliced[k] - 1
#             # print('block', block)
#             # exit(0)
#             alpha = .5
#             ax3.axhspan(times_block_sliced[0], times_block_sliced[-1], facecolor=cs[block], alpha=alpha, label=labels[block])
#             # if block != 6:
#             #     ax3.axhspan(times_block_sliced[0], times_block_sliced[-1], facecolor=cs[block], alpha=alpha, label=labels[block])
#             # else:
#             #     ax3.axhspan(times_block_sliced[0], times_block_sliced[-1], facecolor='gray', alpha=alpha, label='No trans')
#                 
#         #         
#         # exit(0)        
#         ax3.set_ylim(times_sliced_look_ahead[-1], times_sliced_look_ahead[0])
#             # else:
#         # exit(0)
#         # ax3.text(-10, 0, 'Decoder Output (sequence of transition blocks)' + str(decoded_sequence.shape), rotation=90 )
#         ax3.grid()
#         # ax3.set_yticklabels([])
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         plt.legend(by_label.values(), by_label.keys(), loc=1)
#         # print(times_sliced[convolutional_stride::convolutional_stride])
#         
#         # attn_weights_convolved = []
#         # windows_per_conv = conv_w_size//convolutional_stride
#         # num_windows = attention_weights_seq_sliced.shape[1]
#         # for i in range(num_windows + windows_per_conv - 1):
#         #     inds = np.arange(i - windows_per_conv +1, i + 1)
#         #     inds = inds[np.logical_and(inds>= 0, inds <= num_windows - 1)]
#         #     vals = attention_weights_seq_sliced[:, inds]
#         #     attn_weights_convolved.append(np.sum(vals, axis=1)/vals.shape[1])
#         # attn_weights_convolved = np.asarray(attn_weights_convolved).swapaxes(0,1)
#         attn_weights_convolved = convolve_att_weights(conv_w_size, convolutional_stride, attention_weights_seq_sliced)
#         
#         ax2.matshow(attn_weights_convolved, aspect='auto', vmin=0, vmax=1,)
#         
#         ax2.xaxis.set_ticks_position('bottom')
#         ax2.yaxis.set_ticks_position('right')
#         
#         ax2.set_xticks(np.arange(0, attn_weights_convolved.shape[1], 1));
#         ax2.set_xticklabels(np.arange(1, attn_weights_convolved.shape[1] + 1, 1));
#         ax2.set_yticks(np.arange(0, attn_weights_convolved.shape[0], 1));
#         ax2.set_yticklabels(np.arange(1, attn_weights_convolved.shape[0] + 1, 1));
#         ax2.set_xticks(np.arange(-.5, attn_weights_convolved.shape[1], 1), minor=True)
#         ax2.set_yticks(np.arange(-.5, attn_weights_convolved.shape[0], 1), minor=True)
#         # ax2.grid(which='minor')
#         plt.tight_layout()
#         # plt.show()
#         # plt.savefig(fname)
#         fig.savefig(handler, format='pdf')
#         plt.close('all')
#     handler.close()
#     sys.stdout.flush()
    # exit(0)
        
def main():
    shot='33446-ffelici' #30310 45105  33446 32195
    fshot, fshot_times = load_fshot_from_labeler(shot, './data/labeled/')
    gaussian_time_window = 10e-4
    signal_sampling_rate = 1e4
    gaussian_hinterval = int(gaussian_time_window * signal_sampling_rate)
    fshot['sm_elm_label'], fshot['sm_non_elm_label'] = smoothen_elm_values(fshot.ELM_label.values, smooth_window_hsize=gaussian_hinterval)
    fshot['sm_none_label'], fshot['sm_low_label'], fshot['sm_high_label'], fshot['sm_dither_label'] = smoothen_states_values(fshot.LHD_label.values, fshot.time.values, smooth_window_hsize=gaussian_hinterval)
    #CUTOFF to put all elm labels at 0 where state is not high
    fshot.loc[fshot['LHD_label'] != 3, 'ELM_label'] = 0
    fshot = normalize_signals_mean(fshot)
    fshot = state_to_trans_event_disc(fshot, gaussian_hinterval)
    fshot = trans_disc_to_cont(fshot, gaussian_hinterval)
    indexes = np.arange(len(fshot))
    time_window = get_dt_and_time_window_windexes(indexes, fshot)
    transitions = get_transitions_in_window(time_window)
    elms = get_elms_in_window(time_window).swapaxes(0, 1)
    signals = get_raw_signals_in_window(time_window).swapaxes(0, 1)
    times = time_window.time.values
    print(fshot.shape, transitions.shape, elms.shape, signals.shape, times.shape)
    
    # exit(0)
    plot_all_signals_all_trans(elms, signals, times, transitions)
   

    
    
if __name__ == '__main__':
    main()
