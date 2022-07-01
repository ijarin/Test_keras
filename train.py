# general imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from model import AlexnetModel
from data_load import load_cifar10, load_cifar100
#from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#warnings.simplefilter("ignore")
import yaml
import tensorflow as tf
# Load config from YAML file
with open("config.yaml", "r") as fp:
    args = yaml.safe_load(fp)

dataset=args["dataset"]
lr=args["lr"]
batch_size=args["batch_size"]


if dataset=='Cifar10':
  #define all the variables
#cifar10
  shape = (32, 32, 3)
  num_class=10
  learning_rate=.01
  epochs=10
elif dataset=='Cifar100':
  #cifar100
  shape = (32, 32, 3)
  num_class=100
  learning_rate=.01
  epochs=15

#load training/testing data
if dataset=='Cifar10':
       train_data, train_labels, test_data, test_labels = load_cifar10()
elif dataset=='Cifar100':
       train_data, train_labels, test_data, test_labels = load_cifar100()

train_y=to_categorical(train_labels)
test_y=to_categorical(test_labels)

# one model is supposed to train for 10, one for 50 epochs
model_np = AlexnetModel(shape,num_class)
print(model_np.summary())

# specify parameters
#learning_rate=.005
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# compile the model
model_np.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# train the model
print("start training the model....")
history = model_np.fit(train_data, train_y,
                       validation_data=(test_data, test_y),
                       batch_size=batch_size,
                       epochs=epochs)

print("training complete....")
print('Predict on test:Utility Count')
prob_test = model_np.predict(test_data[5000:10000])
#print("model-utility...",prob_test)