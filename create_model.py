from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten
from tensorflow.keras.models import Sequential


def create_model(input_shape, ouput_shape, conv_layers=1, dense_layers=1):
    model = Sequential()

    model.add(Conv2D(10, input_shape=input_shape, kernel_size=(5, 5), strides=(3, 3), activation='relu'))
    for i in range(2, conv_layers + 1):
        model.add(Conv2D(i * 10, kernel_size=(5, 5), strides=(3, 3), activation='relu'))

    model.add(Flatten())

    for i in range(1, dense_layers + 1):
        model.add(Dense(50, activation='relu'))

    model.add(Dense(ouput_shape, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

