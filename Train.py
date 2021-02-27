from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Dropout,Activation, Flatten
from keras.optimizers import SGD
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np


model = Sequential()
model.add(Conv2D(30, (10, 10), input_shape=(100,100, 1), padding='same', activation= "relu"))
model.add(MaxPooling2D((3, 3), strides=2))
model.add(Conv2D(60, (5, 5),activation="relu", strides=1))
model.add(MaxPooling2D((3,3)))
model.add(Conv2D(90, (3, 3),activation="relu", strides=1 ))
model.add(MaxPooling2D((3,3)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer= SGD(lr = lr), metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=10, verbose=1)

class DataGenerator(Sequence):
    def __init__(self,
                 all_filenames,
                 labels,
                 batch_size,
                 index2class,
                 input_dim,
                 n_channels,
                 shuffle = True):
        self.all_filenames = all_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.index2class = index2class
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.all_filenames)/self.batch_size))
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        all_filenames_temp = [self.all_filenames[k] for k in indexes]
        x, y = self.__data_generation(all_filenames_temp)
        return x, y
    def __data_generation(self, all_filenames_temp):
        x = np.empty((self.batch_size))




