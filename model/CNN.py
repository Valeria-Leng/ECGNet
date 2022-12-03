import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation



def buildModel_2DCNN(nb_classes, last_layer='linear'):
    model = tf.keras.models.Sequential()
    # input_xs = tf.keras.Input([5120, 12])
    # convolution layer 1
    model.add(Conv2D(16, kernel_size=(1, 7),
              input_shape=(5120, 12, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))
    # 2
    model.add(Conv2D(16, (1, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='SAME'))
    # 3
    model.add(Conv2D(32, (1, 5), padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))
    # 4
    model.add(Conv2D(32, (1, 5), padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='SAME'))
    # 5
    model.add(Conv2D(64, (1, 5), padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))
    # 6
    model.add(Conv2D(64, (1, 3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))
    # 7
    model.add(Conv2D(64, (1, 3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))
    # 8
    model.add(Conv2D(64, (1, 3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))
    # Single Layer
    model.add(Conv2D(128, (12, 1), padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='SAME'))
    ##################fully connect##############
  # layer1
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.2))
    # layer2
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.2))
    ############output#################
    model.add(Dense(nb_classes, activation= last_layer))
    return model




if __name__ == "__main__":
    model = buildModel_2DCNN(1, last_layer='linear')
    model.summary()