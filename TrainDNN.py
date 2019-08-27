from __future__ import division, print_function, absolute_import

#import tflearn
from tflearn.data_utils import shuffle, to_categorical
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.estimator import regression
# from tflearn.data_preprocessing import ImagePreprocessing
# from tflearn.data_augmentation import ImageAugmentation
from tflearn.datasets import cifar10
import TFLearnDNN as net


# Get the model
model = net.model


# Data loading and preprocessing

(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y,nb_classes=10)
Y_test = to_categorical(Y_test,nb_classes=10)



#Training Module

#model.load("model.tfl")

model.fit(X, Y, n_epoch=15, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=100, run_id='cifar10_cnn')

model.save("model.tfl")
