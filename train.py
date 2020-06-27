
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import glob

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset")
ap.add_argument("-p", "--plot", type = str, default = "plot.png", help = "path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type = str,default = "mask_detector.model", help = "path to output face mask detector model")
args = vars(ap.parse_args())

num_epoch = 10
BS = 32
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

print("[INFO] loading images...")
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

trainData, testData, trainLabel, testLabel = train_test_split(data, labels, test_size = 0.1)

model = Sequential()
model.add(Conv2D(200, (3, 3), input_shape = data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(100, (3, 3), input_shape = data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

print("[INFO] compiling model...")
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#checkpoint = ModelCheckpoint('model/m-{epoch:03d}.model', monitor = 'val_loss', verbose = 0, save_best_only = True)

print("[INFO] training model...")
history = model.fit(trainData, trainLabel, epochs = 10, validation_split = 0.2)

print("[INFO] saving model...")
model.save("model.h5")

print("[INFO] evaluating network...")
predIdxs = model.predict(testData, batch_size=BS)


predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testLabel.argmax(axis=1), predIdxs, target_names=lb.classes_))

N = num_epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

