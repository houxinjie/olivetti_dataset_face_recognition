import numpy as np
np.random.seed(1337)
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.datasets import fetch_olivetti_faces
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
olivetti_faces = fetch_olivetti_faces(data_home=".")
batch_size = 10
nb_classes = 40


# input image dimensions
img_rows, img_cols = 64, 64
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

epochs = 100

X_train = np.empty((320, img_rows * img_cols))
y_train = np.empty(320, dtype=int)

X_validate = np.empty((40, img_rows * img_cols))
y_validate = np.empty(40, dtype=int)

X_test = np.empty((40, img_rows * img_cols))
y_test = np.empty(40, dtype=int)

for i in range(400):
    residue = i % 10
    quotient = int(i / 10)
    if residue < 8:
        X_train[quotient*8+residue] = olivetti_faces.data[i]
        y_train[quotient*8+residue] = olivetti_faces.target[i]
    elif residue < 9:
        X_validate[quotient] = olivetti_faces.data[i]
        y_validate[quotient] = olivetti_faces.target[i]
    else:
        X_test[quotient] = olivetti_faces.data[i]
        y_test[quotient] = olivetti_faces.target[i]

#X_train = X_train / 255
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
y_train = np_utils.to_categorical(y_train, nb_classes)

#X_validate = X_validate / 255
X_validate = X_validate.reshape(X_validate.shape[0], 1, img_rows, img_cols)
y_validate = np_utils.to_categorical(y_validate, nb_classes)

#X_test = X_test / 255
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
y_test = np_utils.to_categorical(y_test, nb_classes)

#model=Sequential([Dense(32,input_dim=4096), Activation('relu'), Dense(40), Activation('softmax')])
#model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])




model = Sequential()
model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), padding='valid', input_shape=(1, img_rows, img_cols), data_format='channels_first'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_validate, y_validate), batch_size=batch_size, epochs=epochs, verbose=1)

loss, accuracy= model.evaluate(X_test, y_test)
print('\nTest lost:', loss)
print('Test accuracy',accuracy)