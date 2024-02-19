
import tensorflow as tf
import os
from matplotlib import pyplot as plt

!pip install opendatasets
import opendatasets as od
od.download("https://www.kaggle.com/datasets/joebeachcapital/defungi")

gpus = tf.config.list_physical_devices('CPU')
print(gpus)

path = os.path.join('Desktop','Shelton','Shelton')

data = tf.keras.utils.image_dataset_from_directory('/content/defungi',
                                                   image_size=(224, 224)
                                                    ,batch_size = 64
                                                   ,shuffle = True)
data = data.map(lambda x, y: (x / 255.0, y))

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

batch[0].shape

train_size = int(len(data)*0.6)
val_size = int(len(data)*0.2)
test_size = int(len(data)*0.2)

train_size,val_size,test_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import layers, regularizers
from tensorflow.keras.regularizers import l2,l1_l2
from tensorflow.keras.applications import ResNet50V2

base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False
keras_model = Sequential()
keras_model.add(base_model)
keras_model.add(Flatten())
keras_model.add(Dropout(0.9))
keras_model.add(Dense(5, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
keras_model.summary()

keras_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist = keras_model.fit(train,
                    epochs=20,
                    validation_data=val)

fig = plt.figure()
plt.plot(hist.history['loss'],color = 'teal',label = 'Loss')
plt.plot(hist.history['val_loss'],color = 'orange',label = 'Val_Loss')
plt.show()

test_loss, test_acc = keras_model.evaluate(test)
