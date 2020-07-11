# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from tensorflow.keras import backend as K

class LeNet:
    @staticmethod
    def build(numChannels, numRows, numCols, numClasses, activation="relu", weightsPath=None):
        # initialize the model
        model = Sequential()
        imageShape = (numRows, numCols, numChannels)

        if K.image_data_format() == "channels_first":
            imageShape = (numChannels, numRows, numCols)

        # define the first set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(20, 5, padding="same", input_shape=imageShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(padding=(2, 2), strides=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(50, 5, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(padding=(2, 2), strides=(2, 2)))

        # define the first FC => ACTIVATION layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # define the second FC => ACTIVATION layers
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
