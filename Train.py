from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Dropout,Activation, Flatten



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


