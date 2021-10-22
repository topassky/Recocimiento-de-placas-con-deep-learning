import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
  def __init__(self, dataFormat=None):
    # store the image data format
    self.dataFormat = dataFormat
    
  def preprocess(self, image):
    # apply the Keras utility function that correctly rearranges
    # the dimensions of the image
    return img_to_array(image, data_format=self.dataFormat)


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors
        # if the preprocessors are None, initialize them as an
        # # empty list
        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,
                    len(imagePaths)))
        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # # ratio
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)



class AspectAwarePreprocessor:
  def __init__(self, width, height, inter=cv2.INTER_AREA):
    # store the target image width, height, and interpolation
    # method used when resizing
    self.width = width
    self.height = height
    self.inter = inter
  
  def preprocess(self, image):
    # grab the dimensions of the image and then initialize
    # the deltas to use when cropping
    (h, w) = image.shape[:2]
    dW = 0
    dH = 0

    # if the width is smaller than the height, then resize
    # along the width (i.e., the smaller dimension) and then
    # update the deltas to crop the height to the desired
    # dimension
    if w < h:
      image = imutils.resize(image, width=self.width,
                             inter=self.inter)
      dH = int((image.shape[0] - self.height) / 2.0)

    # otherwise, the height is smaller than the width so
    # resize along the height and then update the deltas
    # to crop along the width
    else:
      image = imutils.resize(image, height=self.height,
                             inter=self.inter)
      dW = int((image.shape[1] - self.width) / 2.0)

    # now that our images have been resized, we need to
    # re-grab the width and height, followed by performing
    # the crop
    (h, w) = image.shape[:2]
    image = image[dH:h - dH, dW:w - dW]

    # finally, resize the image to the provided spatial
    # dimensions to ensure our output image is always a fixed
    # size
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, (self.width, self.height),
                      interpolation=self.inter)

# import the necessary packages
from tensorflow.keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
  def __init__(self, outputPath, every=5, startAt=0):
    # call the parent constructor
    super(Callback, self).__init__()

		# store the base output path for the model, the number of
		# epochs that must pass before the model is serialized to
		# disk and the current epoch value
    self.outputPath = outputPath
    self.every = every
    self.intEpoch = startAt
  def on_epoch_end(self, epoch, logs={}):
    # check to see if the model should be serialized to disk
    if (self.intEpoch + 1) % self.every == 0:
      p = os.path.sep.join([self.outputPath,
        "epoch_{}.hdf5".format(self.intEpoch + 1)])
      self.model.save(p, overwrite=True)
    # increment the internal epoch counter
    self.intEpoch += 1

# import the necessary packages
from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
  def __init__(self, figPath, jsonPath=None, startAt=0):
    # store the output path for the figure, the path to the JSON
    # serialized file, and the starting epoch
    super(TrainingMonitor, self).__init__()
    self.figPath = figPath
    self.jsonPath = jsonPath
    self.startAt = startAt

  def on_train_begin(self, logs={}):
    # initialize the history dictionary
    self.H = {}
    # if the JSON history path exists, load the training history
    if self.jsonPath is not None:
      if os.path.exists(self.jsonPath):
        self.H = json.loads(open(self.jsonPath).read())
        # check to see if a starting epoch was supplied
        if self.startAt > 0:
          # loop over the entries in the history log and
          # trim any entries that are past the starting
          # epoch
          for k in self.H.keys():
            self.H[k] = self.H[k][:self.startAt]
  
  def on_epoch_end(self, epoch, logs={}):
    # loop over the logs and update the loss, accuracy, etc.
    # for the entire training process
    for (k, v) in logs.items():
      l = self.H.get(k, [])
      l.append(v)
      self.H[k] = l
    # check to see if the training history should be serialized 
    # to file
    if self.jsonPath is not None:
      f = open(self.jsonPath, "w")
      f.write(json.dumps(self.H))
      f.close()
    # ensure at least two epochs have passed before plotting
    # (epoch starts at zero)
    if len(self.H["loss"]) > 1:
      # plot the training loss and accuracy
      N = np.arange(0, len(self.H["loss"]))
      plt.style.use("ggplot")
      plt.figure()
      plt.plot(N, self.H["loss"], label="train_loss")
      plt.plot(N, self.H["val_loss"], label="val_loss")
      plt.plot(N, self.H["accuracy"], label="train_acc")
      plt.plot(N, self.H["val_accuracy"], label="val_acc")
      plt.title("Training Loss and Accuracy [Epoch {}]".format(
          len(self.H["loss"])))
      plt.xlabel("Epoch #")
      plt.ylabel("Loss/Accuracy")
      plt.legend()
    # save the figure
    plt.savefig(self.figPath)
    plt.close()
