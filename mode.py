from keras_model import Keras
from preprocessing import Preprocessing
import numpy as np


class Mode:

    def __init__(self, mode, datadir=None, print_=False):
        
        self.mode = mode

        self.__dataset__(datadir)

        if print_ == True:
            self.print_mode()


    def get_info_trained(self):

        if self.mode == 1 or self.mode == 2:
            return "Mode_{}".format(self.mode)
        else:
            return "Final_mode"


    def get_mode(self):
        return self.mode
    

    def print_mode(self):
        prog_mode = self.__info_mode__()
        print("The chosen mode is {}.\n{}".format(self.mode, prog_mode))


    def __info_mode__(self):

        if self.mode == 1:
            return "The mode 1 runs the training process without augmentation and the applied sets are train set and valid set."
        elif self.mode == 2:
            return "The mode 2 runs the training process with augmentation and the applied sets are train set and valid set."
        else:
            return "The final mode (3) runs the training process without augmentation and the applied sets are train set and test set."


    def get_NN(self):
        return self.NN


    def run_mode(self, batch_=256, iteration_=50, verb_=1):
        self.NN = Keras(self.data_1, self.data_2, self.labels_1, self.labels_2)

        if self.mode == 1 or self.mode == 3:
            self.NN.train_network(batch=batch_, iteration=iteration_, verb=verb_)
        else:
            self.NN.train_network_with_augmentation(batch=batch_, iteration=iteration_, verb=verb_)


    def data_info(self):
        print("Information by dataset:")

        print('Training data:', self.data_1.shape, self.labels_1.shape)
        print("Testing data:" if self.mode == 3 else 'Validating data:', self.data_2.shape, self.labels_2.shape)
        
        np_labels = np.unique(self.labels_1)
        len_labels = len(np_labels)

        print('Labels:', np_labels)
        print('Total labels:', len_labels)


    def __dataset__(self, datadir):
        dataset = Preprocessing(datadir)
        dataset.get_normalized_data()
        dataset.reshape_data()
        data = dataset.get_data()
        labels = dataset.get_labels()

        (train, validating, train_l, validating_l) = dataset.split_data(data, labels, size_of_test=0.2, rand_state=None)
        (valid, test, valid_l, test_l) = dataset.split_data(validating, validating_l, size_of_test=0.5, rand_state=None)

        if self.mode == 1 or self.mode == 2:
            self.data_1 = train
            self.data_2 = valid
            self.labels_1 = train_l
            self.labels_2 = valid_l
        else:
            self.data_1 = train
            self.data_2 = test
            self.labels_1 = train_l
            self.labels_2 = test_l