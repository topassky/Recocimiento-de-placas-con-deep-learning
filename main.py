# python main.py --dataset dataset --checkpoints output/checkpoints 

#python main.py --dataset dataset --checkpoints output/checkpoints --model output/checkpoints/epoch_20.hdf5 --start-epoch 20

# import the necessary packages
#from tensorflow import keras 
#from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from comcop.load_dataset import load_dataser_config
from comcop.nn.resnet import ResNet
from comcop.nn.lenet import LeNet
from comcop.nn.vgg import EmotionVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import argparse
import os

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
    help="dataset")
ap.add_argument("-c", "--checkpoints", required=True,
    help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
    help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
    help="epoch to restart training at")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32


# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

aap = load_dataser_config.AspectAwarePreprocessor(28, 28)
sdl = load_dataser_config.SimpleDatasetLoader(preprocessors=[aap])
(data, labels) = sdl.load(imagePaths, verbose=100)


# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# if there is no specific model checkpoint supplied, then initialize
# the network (ResNet-56) and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    model = LeNet.build(width=28, height=28, depth=3, classes=2)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
# otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [
    load_dataser_config.EpochCheckpoint(args["checkpoints"], every=5,
        startAt=args["start_epoch"]),
    load_dataser_config.TrainingMonitor("output/resnet56_placas.png",
        jsonPath="output/resnet56_placas.json",
    startAt=args["start_epoch"])]

# train the network
print("[INFO] training network...")
model.fit(
    x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), 
    steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, 
    callbacks=callbacks,
    verbose=1)



    

    
