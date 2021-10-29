# BGP Anomalies Classification using Features based on AS Relationship Graphs


This project contains the implementation of the LSTM-based classification of BGP anomalies
based on our paper

> T. B. Paiva, Y. Siqueira, D. M. Batista, R. Hirata Jr. and R. Terada. BGP Anomalies Classification using
Features based on AS Relationship Graphs. Proceedings of the IEEE Latin-American Conference on Communications
(to appear).

The paper can be found in the author's archive [here](https://www.ime.usp.br/~tpaiva/).


# Structure

* `anomalies.csv`
    CSV containing information on anomalous events.
* `asn_countries.csv`
    CSV containing data on the countries and approximate geographical location of ASes.
* `classifier.py`
    Implements the LSTM classifier of BGP anomalies.
* `data.zip`
    Compressed file containing the processed data: features and labels ready for training our model.
* `data_collector.py`
    Program to download BGP updates corresponding to events given in a CSV.
* `extract_features.py`
    Program to extract features from BGP updates.
* `paths_collector.py`
    Program to extract collect paths from two days before the events that allows us to
    run AS relationship inference algorithms.
* `relationships_inference.csv`
    Implements Lixin Gao procedure for classifying Valley-Free paths and AS relationship inference.

# Usage

In this section, we describe the steps to reproduce the results of our paper.

**Before we start**, please notice that module `pybgpstream`,  that we use for downloading BGP
data, requires that [CAIDA's library `libBGPStream`](https://bgpstream.caida.org/) is installed. This can be done following
the [instructions given on the projects page](https://bgpstream.caida.org/docs/install/bgpstream). You must install
the library before proceeding to the next step.

**However, you do not need to install `pybgpstream` if you use our data compressed in `data.zip`**


First let us clone the repository and `cd` to it
```
$ git clone https://github.com/thalespaiva/bgp-anomaly-classification.git
$ cd bgp-anomaly-classification
```

Now we install our environment and the project's dependencies using Pipenv.
```
$ pipenv shell
$ pipenv install
```

We are ready to use the code :^)

## Data collection

    Todo.


## Running the LSTM

```
$ pipenv shell
$ unzip data
Archive:  data.zip
   creating: data/
   creating: data/ready/
  inflating: data/ready/test.features
  inflating: data/ready/train-val.features
   creating: data/events/
  inflating: data/events/AS9121-RTL.ris.rrc05.updates.features
  inflating: data/events/AWS-Route-Leak.ris.rrc04.updates.features
  inflating: data/events/Code-Red-v2.ris.rrc04.updates.features
  inflating: data/events/Japan-Earthquake.ris.rrc06.updates.features
  inflating: data/events/Malaysian-Telecom.ris.rrc04.updates.features
  inflating: data/events/Moscow-Blackout.ris.rrc05.updates.features
  inflating: data/events/Nimda.ris.rrc04.updates.features
  inflating: data/events/Slammer.ris.rrc04.updates.features
  inflating: data/events/WannaCrypt.ris.rrc04.updates.features
$ ./classifier.py data/ready/train-val.features data/ready/test.features out.model  --run-test-events-seen --run-test-events-not-seen --only-anomalous
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d (Conv1D)              (None, 10, 32)            1120
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 5, 32)             0
_________________________________________________________________
lstm (LSTM)                  (None, 100)               53200
_________________________________________________________________
dropout (Dropout)            (None, 100)               0
_________________________________________________________________
dense (Dense)                (None, 3)                 303
=================================================================
Total params: 54,623
Trainable params: 54,623
Non-trainable params: 0
_________________________________________________________________
None
class=0 => len(x_train)=14, len(x_valid)=4, len(x_test)=3, len(xs_ftest)=11
class=1 => len(x_train)=353, len(x_valid)=101, len(x_test)=52, len(xs_ftest)=99
class=2 => len(x_train)=366, len(x_valid)=105, len(x_test)=54, len(xs_ftest)=138
Epoch 1/10
366/366 [==============================] - 2s 3ms/step - loss: 0.6133 - categorical_accuracy: 0.8858 - val_loss: 0.1989 - val_categorical_accuracy: 0.9619
Epoch 2/10
366/366 [==============================] - 1s 2ms/step - loss: 0.1378 - categorical_accuracy: 0.9709 - val_loss: 0.2822 - val_categorical_accuracy: 0.9048
Epoch 3/10
366/366 [==============================] - 1s 2ms/step - loss: 0.0939 - categorical_accuracy: 0.9650 - val_loss: 0.0831 - val_categorical_accuracy: 0.9905
Epoch 4/10
366/366 [==============================] - 1s 2ms/step - loss: 0.0031 - categorical_accuracy: 1.0000 - val_loss: 0.0920 - val_categorical_accuracy: 0.9905
Epoch 5/10
366/366 [==============================] - 1s 2ms/step - loss: 0.0013 - categorical_accuracy: 1.0000 - val_loss: 0.0989 - val_categorical_accuracy: 0.9905
Epoch 6/10
366/366 [==============================] - 1s 2ms/step - loss: 5.8327e-04 - categorical_accuracy: 1.0000 - val_loss: 0.1038 - val_categorical_accuracy: 0.9905
Epoch 7/10
366/366 [==============================] - 1s 2ms/step - loss: 5.3993e-04 - categorical_accuracy: 1.0000 - val_loss: 0.1079 - val_categorical_accuracy: 0.9905
Epoch 8/10
366/366 [==============================] - 1s 2ms/step - loss: 2.5287e-04 - categorical_accuracy: 1.0000 - val_loss: 0.1115 - val_categorical_accuracy: 0.9905
Epoch 9/10
366/366 [==============================] - 1s 2ms/step - loss: 1.0871e-04 - categorical_accuracy: 1.0000 - val_loss: 0.1141 - val_categorical_accuracy: 0.9905
Epoch 10/10
366/366 [==============================] - 1s 2ms/step - loss: 9.7036e-05 - categorical_accuracy: 1.0000 - val_loss: 0.1168 - val_categorical_accuracy: 0.9905
2021-10-29 00:20:45.171132: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
========== VALIDATION REPORT===============
Confusion matrix
tf.Tensor(
[[ 4  0  0]
 [ 0 97  0]
 [ 0  1  3]], shape=(3, 3), dtype=int32)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         4
           1       0.99      1.00      0.99        97
           2       1.00      0.75      0.86         4

    accuracy                           0.99       105
   macro avg       1.00      0.92      0.95       105
weighted avg       0.99      0.99      0.99       105

========== TEST (SPLIT SET) REPORT===============
Confusion matrix
tf.Tensor(
[[ 3  0  0]
 [ 0 49  0]
 [ 0  0  2]], shape=(3, 3), dtype=int32)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         3
           1       1.00      1.00      1.00        49
           2       1.00      1.00      1.00         2

    accuracy                           1.00        54
   macro avg       1.00      1.00      1.00        54
weighted avg       1.00      1.00      1.00        54

========== TEST (UNSEEN FEATURES) REPORT===============
Confusion matrix
tf.Tensor(
[[11  0  0]
 [ 0 88  0]
 [ 0 13 26]], shape=(3, 3), dtype=int32)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       0.87      1.00      0.93        88
           2       1.00      0.67      0.80        39

    accuracy                           0.91       138
   macro avg       0.96      0.89      0.91       138
weighted avg       0.92      0.91      0.90       138
```


# License

MIT


# Authors

* Thales Paiva
* Yaissa Siqueira
* Daniel Macêdo Batista
* Roberto Hirata
* Routo Terada