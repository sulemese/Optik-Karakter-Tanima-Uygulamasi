from keras import layers
from keras.models import Model
from keras.layers import LSTM
import keras.saving
from mltu.tensorflow.model_utils import residual_block


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    
    #Modelin giriş katmanı, 32x128 boyutlarında ve 3 kanallı bir RGB görüntü alır. 
    inputs = layers.Input(shape=input_dim, name="input")
    #Lambda katmanı kullanılarak giriş görüntülerinin piksel değerleri, [0, 255] aralığından [0, 1] aralığına ölçeklenir.
    input = layers.Lambda(lambda x: x / 255)(inputs)
    #resnetten esinlenilmiştir. 
    x1 = residual_block(input, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x2 = residual_block(x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = residual_block(x5, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x7 = residual_block(x6, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    squeezed = layers.Reshape((x7.shape[-3] * x7.shape[-2], x7.shape[-1]))(x7)

    blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(squeezed)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model