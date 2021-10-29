#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback


# Sets global seeds to ensure reproducibility
from numpy.random import seed
seed(1)
tf.random.set_seed(2)


class LRFinder(Callback):
    
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''
    
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        
    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations 
        return self.min_lr + (self.max_lr-self.min_lr) * x
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        K.set_value(self.model.optimizer.lr, self.clr())
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()


# Features to use for classification
FEATURES = [
    'max_as_path_length',
    'av_as_path_length',
    'av_number_of_bits_in_prefix_ipv4',
    'max_number_of_bits_in_prefix_ipv4',
    'max_unique_as_path_length',
    'av_unique_as_path_length',
    'var_as_degree_in_paths',
    'av_as_degree_in_paths',
    'av_number_of_edges_not_in_as_graph',
    'av_number_of_P2P_edges',
    'av_number_of_C2P_edges',
    'av_number_of_P2C_edges',
    'av_number_of_S2S_edges',
    'av_number_of_non_vf_paths',
    'avg_geo_dist_same_bitlen',
    'avg_geo_dist_diff_bitlen',
    'n_announcements'
]

N_CLASSES = 3
BATCH_SIZE = 1
EPOCHS = 10
SEQUENCE_LENGTH = 10
TRAIN_SIZE = 0.7
TEST_SIZE= 0.1

SEED_SPLIT1 = 0
SEED_SPLIT2 = 1

def train_valid_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE):
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=SEED_SPLIT1)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=test_size/(1 - train_size), random_state=SEED_SPLIT2)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_sequences(df, sequence_length, overlapping_sequences=False):
    Xs, ys = {}, {}

    X = df[FEATURES]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    if not overlapping_sequences:
        step = sequence_length
    else:
        step = 1

    for i in range(0, len(X_scaled) - sequence_length, step):
        y = np.zeros(N_CLASSES)
        label_index = max(df.label[i:i + sequence_length])
        y[label_index] = 1

        if label_index not in Xs:
            Xs[label_index] = []
            ys[label_index] = []

        Xs[label_index].append(X_scaled[i:i + sequence_length])
        ys[label_index].append(y)
    
    return Xs, ys


def get_model(n_features, n_sequence):

    model = keras.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=2, input_shape=(n_sequence, n_features), padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.LSTM(100))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(N_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0067),
                  metrics=['categorical_accuracy'])

    return model

def get_classification_report(model, xs, ys):
    predictions = np.argmax(model.predict(np.array(xs)), axis=-1)
    labels = [np.argmax(y) for y in ys]

    cm = tf.math.confusion_matrix(labels=labels, predictions=predictions)

    print('Confusion matrix')
    print(cm)

    print(classification_report(labels, predictions))


def main():

    df = pd.read_csv(args.features).fillna(0)
    df_final_test = pd.read_csv(args.separated_features).fillna(0)

    if (args.only_anomalous):
        df = df[df.label != -1]

    Xs, ys = get_sequences(df, args.sequence_length, overlapping_sequences=args.overlapping_sequences)
    Xs_f, ys_f = get_sequences(df_final_test, args.sequence_length, overlapping_sequences=False)

    model = get_model(len(FEATURES), args.sequence_length)
    print(model.summary())

    x_train, x_valid, x_test, y_train, y_valid, y_test = [], [], [], [], [], []
    xs_ftest, ys_ftest = [],[]
    for key in Xs:
        split = train_valid_test_split(Xs[key], ys[key], args.train_size, args.test_size)
        x_temp_train, x_temp_valid, x_temp_test, y_temp_train, y_temp_valid, y_temp_test = split

        x_train.extend(x_temp_train)
        x_valid.extend(x_temp_valid)
        x_test.extend(x_temp_test)
        y_train.extend(y_temp_train)
        y_valid.extend(y_temp_valid)
        y_test.extend(y_temp_test)

        xs_ftest.extend(Xs_f[key])
        ys_ftest.extend(ys_f[key])

        print(f'class={key} => {len(x_train)=}, {len(x_valid)=}, {len(x_test)=}, {len(xs_ftest)=}')

    
    xy_train = list(zip(x_train, y_train))
    np.random.shuffle(xy_train)
    x_train, y_train = zip(*xy_train)

    if args.calculate_learning_rate:
        lr_finder = LRFinder(min_lr=1e-5, 
                                    max_lr=1e-2, 
                                    steps_per_epoch=np.ceil(EPOCHS/BATCH_SIZE), 
                                    epochs=3)
        
        model.fit(
            np.asarray(x_train),
            np.asarray(y_train),
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(np.array(x_valid), np.array(y_valid)),
            callbacks=[lr_finder])
        
        lr_finder.plot_loss()
    else:
        model.fit(
            np.asarray(x_train),
            np.asarray(y_train),
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(np.array(x_valid), np.array(y_valid)))
    
    model.save(args.output)

    print("========== VALIDATION REPORT===============")
    get_classification_report(model, x_valid, y_valid)

    if (args.run_test_events_seen):
        print("========== TEST (SPLIT SET) REPORT===============")
        get_classification_report(model, x_test, y_test)

    if (args.run_test_events_not_seen):
        print("========== TEST (UNSEEN FEATURES) REPORT===============")
        get_classification_report(model, xs_ftest, ys_ftest)
   
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('features', help='Path to file containing features and labels.')
    parser.add_argument('separated_features', help='Path to file containing separated features to a final test and its labels.')
    parser.add_argument('output', help='Path where the LSTM model will be saved.')
    parser.add_argument('--run-test-events-seen', action='store_true',
                        help='Flag to run classification on test set and show results.')
    parser.add_argument('--run-test-events-not-seen', action='store_true',
                        help='Flag to run classification on test set of unseen events and show results.')
    parser.add_argument('--sequence-length', default=SEQUENCE_LENGTH, type=int,
                        help=f'Length of the sequence fed to the LSTM network (default {SEQUENCE_LENGTH}).')
    parser.add_argument('--train-size', type=float, default=TRAIN_SIZE, help=f'Fraction used for training (default {TRAIN_SIZE}).')
    parser.add_argument('--test-size', type=float, default=TEST_SIZE, help=f'Fraction used for testing (remaining value will be used for validation set) (default {TEST_SIZE}).')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help=f'Number of epochs (default {EPOCHS}).')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size (default {BATCH_SIZE}).')
    parser.add_argument('--overlapping-sequences', action='store_true', help='Flag to consider overlapping sequences.')
    parser.add_argument('--only-anomalous', action='store_true', help='Flag to consider only anomalous data for'
                                                                      'classification.')
    parser.add_argument('--calculate-learning-rate', action='store_true', help='Flag to plot a learning rate comparison graph.')
    args = parser.parse_args()

    main()
