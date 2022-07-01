from tensorflow import keras
from keras.datasets import cifar10
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np



#Load cifar10 dataset
def load_cifar10():
    """Loads cifar10-Dataset and preprocesses to combine training and test data."""

    # load the existing CIFAR10 dataset that comes in form of traing + test data and labels
    train, test = tf.keras.datasets.cifar10.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    # scale the images from color values 0-255 to numbers from 0-1 to help the training process
    train_data1 = np.array(train_data, dtype=np.float32) / 255
    test_data1 = np.array(test_data, dtype=np.float32) / 255

    # cifar10 labels come one-hot encoded, there
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return train_data, train_labels, test_data, test_labels

#load cifar100 dataset
def load_cifar100():
    """Loads cifar100-Dataset and preprocesses to combine training and test data."""

    # load the existing CIFAR10 dataset that comes in form of traing + test data and labels
    train, test = tf.keras.datasets.cifar100.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    # scale the images from color values 0-255 to numbers from 0-1 to help the training process
    train_data1 = np.array(train_data, dtype=np.float32) / 255
    test_data1 = np.array(test_data, dtype=np.float32) / 255

    # cifar10 labels come one-hot encoded, there
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return  train_data, train_labels, test_data, test_labels



