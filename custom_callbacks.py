import keras
from keras.callbacks import Callback
import numpy as np
import pickle
from helper_funcs import load_dic, remove_current_1MA, remove_current_30kA, remove_current_370kA, remove_no_state, remove_disruption_points, normalize_current_MA, normalize_signals_mean, calc_mode, k_statistic
import pandas as pd
from collections import defaultdict
from glob import glob
import resource
import gc
import json
import utilDANNModel
from sklearn.metrics import accuracy_score

class ValidationKappaScore(Callback):
    """
    ValidationKappaScore computation callback.
    Computes the kappa score in the validation set 
    on epoch end.
    """
    def __init__(self, exp, machine_ids, normalization, arch_with_elms, config):
        """
        Args:
        exp: experiment number.
        machine_ids: list of machine where the model will
        be evaluated. e.g ['TCV','JET'].
        normalization: normalization function applied to the signals: e.g minmax
        """
        super().__init__()
        
        self.test_set_aug = [35248, 35274, 35275, 35529, 35530, 35531, 35532, 35536, 35538, 35540, 35556, 35557, 35564, 35582, 35584, 35604, 35607, 35837, 35852, 35967, 35972, 35975] #35537, 35539, 35561] removed for the moment because no DML
        self.test_set_jet = [97927, 97803, 96532, 97828, 97830, 97832, 97835, 97971, 97465, 97466, 97468, 97469, 97473, 94785, 97476, 97477, 96713, 97745, 98005, 96993, 96745, 94315, 97396, 96885, 97398, 97399, 94968, 94969, 94971, 94973]
        #self.test_set_jet = [87871, 87875, 91606, 91468, 91470, 91666, 91118, 91123, 91125, 94126, 96293, 81234, 94028, 85956, 81212, 85897, 81206, 96300, 94032, 82228, 95312, 87539, 81883, 94114, 97974, 91594, 91597, 91605]
        self.test_set_tcv = [61057, 64770, 57093, 64774, 57095, 61714, 64662, 62744, 59065, 64060, 59073, 60097, 61010, 61274, 58460, 61021, 64369, 61043]

        self.exp = exp
        self.model_dir = 'experiments/' + self.exp
        self.machine_ids = machine_ids
        self.normalization = normalization
        self.arch_with_elms = arch_with_elms
        self.labelers = ['marceca']
        self.shots = defaultdict(list)
        self.config = config
        self.conv_window_size = 40
        self.conv_w_offset = 20
        self.no_input_channels = 3
        self.X_scalars_test = defaultdict(list)
        self.GT = defaultdict(list)
        self.fshots = defaultdict(list)
        self.states_pred_concat = defaultdict(list)
        self.ground_truth_concat = defaultdict(list)
        self.avg_kappa = defaultdict(list)
        self.per_shot_pred = defaultdict(dict)
        self.per_shot_gt = defaultdict(dict)
        self.per_shot_kappa = defaultdict(dict)
        
        for machine in self.machine_ids:
            
            if machine == 'TCV':
                self.shots[machine] = self.test_set_tcv
            elif machine == 'JET':
                self.shots[machine] = self.test_set_jet
            elif machine == 'AUG':
                self.shots[machine] = self.test_set_aug
             
            for i, shot in zip(range(len(self.shots[machine])), self.shots[machine]):
                fshot = pd.read_csv('./labeled_data/' + machine + '/' + self.labelers[0] + '/' + machine + '_'  + str(shot) + '_' + self.labelers[0] + '_labeled.csv', encoding='utf-8')
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
                shot_df = normalize_signals_mean(shot_df, machine, func=self.normalization)
                
                stride=10
                length = int(np.ceil((len(shot_df)-self.conv_window_size)/stride))
                X_scalars_single = np.empty((length, self.conv_window_size, self.no_input_channels))
                intersect_times = np.round(shot_df.time.values,5)
                intersect_times = intersect_times[self.conv_window_size-self.conv_w_offset:len(intersect_times)-self.conv_w_offset]
                for j in np.arange(length):
                    vals = shot_df.iloc[j*stride : self.conv_window_size + j*stride]
                    scalars = np.asarray([vals.GWfr, vals.PD, vals.WP]).swapaxes(0, 1)
                    assert scalars.shape == (self.conv_window_size, self.no_input_channels)
                    X_scalars_single[j] = scalars
                self.X_scalars_test[machine] += [X_scalars_single]
                fshot_sliced = shot_df.loc[shot_df['time'].round(5).isin(intersect_times)]
                self.GT[machine] += [fshot_sliced['LHD_label'].values[0::stride]]
                self.fshots[machine] += [fshot_sliced]
                pass
            pass
    def predict(self):
        
        self.weights = self.model.get_weights()
        dann = utilDANNModel.DANNModel(self.config.model, (None, 40, 3), 3, self.config.batch_size)
        self.Mymodel = dann.dann_model
        self.Mymodel.set_weights(self.weights)
        
        names = [weight.name for layer in self.Mymodel.layers for weight in layer.weights]
        weights = self.Mymodel.get_weights()
        
        for machine in self.machine_ids:
            #print('Predicting on machine {}'.format(machine))
            for s_ind, s in enumerate(self.shots[machine]):
                self.Mymodel.reset_states()
                domain, states = self.Mymodel.predict(np.asarray([self.X_scalars_test[machine][s_ind][:, :, :]]), batch_size=1, verbose=0)
                labeler_states = np.asarray(self.GT[machine][s_ind])
                labeler_states = np.expand_dims(labeler_states, axis=0)
                pred_states_disc = np.argmax(states[0,:], axis=1)
                pred_states_disc = pred_states_disc[:len(self.fshots[machine][s_ind])]
                pred_states_disc += 1 #necessary because argmax returns 0 to 2, while we want 1 to 3!

                self.per_shot_pred[machine][s] = pred_states_disc
                self.states_pred_concat[machine].extend(pred_states_disc)

                assert(labeler_states.shape[1] == pred_states_disc.shape[0])
                ground_truth = calc_mode(labeler_states.swapaxes(0,1))
                self.ground_truth_concat[machine].extend(ground_truth)
                self.per_shot_gt[machine][s] = ground_truth
            pass

    def compute_kappa(self):
        for machine in self.machine_ids:
            self.avg_kappa[machine] = k_statistic(np.asarray(self.states_pred_concat[machine]), np.asarray(self.ground_truth_concat[machine]))
            for s_ind, s in enumerate(self.shots[machine]):
                self.per_shot_kappa[machine][s] = k_statistic(np.asarray(self.per_shot_pred[machine][s]), np.asarray(self.per_shot_gt[machine][s]))

    def compute_accuracy(self):
        pred = []
        GT = []
        for machine in self.machine_ids:
            pred.extend(self.states_pred_concat_domain[machine])
            GT.extend(self.ground_truth_concat_domain[machine])
        return accuracy_score(GT, pred)

    def on_epoch_end(self, epoch, logs={}):
        #print('memory usage', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Predict
        self.predict()
        self.compute_kappa()
        #acc = self.compute_accuracy()
        # Log results
        myfile_kappa = open(self.model_dir + '/kappa_scores_exp_{}_epoch_{}.txt'.format(self.exp, epoch), 'w')
        with open(self.model_dir + '/acc_and_loss_exp_{}.txt'.format(self.exp), 'a+') as myfile_loss:
            myfile_loss.seek(0)
            data = myfile_loss.read(100)
            if len(data) > 0 :
                myfile_loss.write("\n")
            myfile_loss.write(json.dumps(logs))

        for machine in self.machine_ids:
            myfile_kappa.write("%s\n" % machine)
            for s_ind, s in enumerate(self.shots[machine]):
                myfile_kappa.write("shot: {}, kappa: {}\n".format(s, self.per_shot_kappa[machine][s]))
            myfile_kappa.write("%s\n" % np.array2string(self.avg_kappa[machine]))
            print("On epoch {}, kappa for machine {} is: {}.".format(epoch, machine, self.avg_kappa[machine]))
            pass
        #print("On epoch {}, domain acc is: {}.".format(epoch, acc))

        myfile_kappa.close()
        #keras.backend.clear_session()
        del self.Mymodel
        gc.collect()
        # Release objects
        #self.states_pred_concat_domain = defaultdict(list)
        self.states_pred_concat = defaultdict(list)
        self.ground_truth_concat = defaultdict(list)
        #self.ground_truth_concat_domain = defaultdict(list)
        self.avg_kappa = defaultdict(list)
