# -*- coding: utf-8 -*-
import utilModels
import numpy as np
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Lambda, TimeDistributed
from keras import backend as K

class NNModel(object):
    # -------------------------------------------------------------------------
    def __init__(self, model_number, input_shape, nb_classes, batch_size, grl='auto', summary=False):
        self.learning_phase = K.variable(1)
        self.model_number = model_number
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.batch_size = batch_size

        #self.opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.opt = optimizers.Adam()

        self.clsModel = getattr(utilModels, "ModelV" + str(model_number))(input_shape)

        self.dann_model, self.label_model, self.tsne_model = self.__build_dann_model()

        self.compile()

    # -------------------------------------------------------------------------
    def load(self, filename):
        weight = np.load(filename, allow_pickle=True)
        self.dann_model.set_weights(weight)

    # -------------------------------------------------------------------------
    def save(self, filename):
        np.save(filename, self.dann_model.get_weights())

    # -------------------------------------------------------------------------
    def compile(self):
        self.dann_model.compile(loss={'out_states': 'categorical_crossentropy'},optimizer=self.opt,metrics=['categorical_accuracy'])

        self.label_model.compile(loss='categorical_crossentropy',
                                                                 optimizer=self.opt,
                                                                 metrics=['categorical_accuracy'])

        self.tsne_model.compile(loss='categorical_crossentropy',
                                                                optimizer=self.opt,
                                                                metrics=['categorical_accuracy'])


    # -------------------------------------------------------------------------
    def __build_dann_model(self):
        branch_features = self.clsModel.get_model_features()

        # Build label model...
        # When building DANN model, route first half of batch (source examples)
        # to domain classifier, and route full batch (half source, half target)
        # to the domain classifier.
        # Build label model...
        branch_label = self.clsModel.get_model_labels(branch_features)
        #branch_label = Dense(self.nb_classes, activation='softmax', name='classifier_output')(branch_label)
        branch_label = TimeDistributed(Dense(self.nb_classes, activation='softmax'), name='out_states')(branch_label)

        # Create models...
        dann_model = Model(input=self.clsModel.input, output=[branch_label])
        label_model = Model(input=self.clsModel.input, output=branch_label)
        tsne_model = Model(input=self.clsModel.input, output=branch_features)

        print(dann_model.summary())

        return dann_model, label_model, tsne_model
