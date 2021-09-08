# -*- coding: utf-8 -*-
import abc
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, TimeDistributed, LSTM
from keras.layers import GlobalAveragePooling2D
from keras import backend as K


# ----------------------------------------------------------------------------
class AbstractModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_shape):
        self.input = Input(shape=input_shape, name="input_signals")

    @abc.abstractmethod
    def get_model_features(self):
        pass

    @abc.abstractmethod
    def get_model_labels(self, input):
        pass

    @abc.abstractmethod
    def get_model_domains(self, input):
        pass

# ----------------------------------------------------------------------------
# v99 - ConvLSTM model
class ModelV99(AbstractModel):
    def get_model_features(self):

        conv_input = Input(shape=(40, 3,), dtype='float32', name='conv_input')
        net = Conv1D(32, 3, activation='relu', padding='same')(conv_input)
        net = Conv1D(64, 3, activation='relu', padding='same')(net)
        net = Dropout(.5)(net)
        net = MaxPooling1D(2)(net)
        net = Flatten()(net)
        net = Dense(16, activation='relu')(net)
        modelCNN = Model(inputs=[conv_input], outputs= [net])

        print('self.input: ', self.input)
        net = TimeDistributed(modelCNN)(self.input)
        net = LSTM(32, return_sequences=True, stateful=False)(net)
        return net

    def get_model_labels(self, input):
        net = TimeDistributed(Dense(8, activation='relu'))(input)
        net = Dropout(.5)(net)
        return net

    def get_model_domains(self, input):
        net = Dense(128, activation='relu')(input)
        net = Dropout(0.5)(net)
        return net
