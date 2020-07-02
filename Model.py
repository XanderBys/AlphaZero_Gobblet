from keras.models import Model as keras_model
from keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
from loss import softmax_crossentropy
import config

class Model:
    def __init__(self, num_states, num_actions, hidden_layers, reg_const, learning_rate):
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_layers = hidden_layers
        self.num_hidden = len(hidden_layers)
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        
        self.nn = None
        
        self.define_model()
    
    def define_model(self):
        inp = Input(shape=self.num_states, name='input')
 
        x = self.conv_layer(inp, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])
        
        if len(self.hidden_layers) > 1:
            for layer in self.hidden_layers[1:]:
                x = self.residual_layer(x, layer['filters'], layer['kernel_size'])
        
        value = self.value_head(x)
        policy = self.policy_head(x)
        
        self.nn = keras_model(inputs=[inp], outputs=[value, policy])
        self.nn.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_crossentropy},
                        optimizer=SGD(lr=self.learning_rate, momentum=config.MOMENTUM), loss_weights={'value_head':0.5, 'policy_head':0.5})
    
    def value_head(self, x):
        x = Conv2D(filters=1, kernel_size=(1,1), data_format="channels_first", padding='same',
                   use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        
        x = Flatten()(x)
        
        x = Dense(20, use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = LeakyReLU()(x)
        
        x = Dense(1, use_bias=False, activation='tanh', kernel_regularizer=regularizers.l2(self.reg_const), name='value_head')(x)
        
        return (x)
    
    def policy_head(self, x):
        x = Conv2D(filters=2, kernel_size=(1,1), data_format="channels_first", padding='same',
                   use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        
        x = Flatten()(x)
        
        x = Dense(20, use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = LeakyReLU()(x)
        
        x = Dense(self.num_actions, use_bias=False, activation='tanh', kernel_regularizer=regularizers.l2(self.reg_const), name='policy_head')(x)
        
        return (x)
    
    def residual_layer(self, input_block, filters, kernel_size):
        x = self.conv_layer(input_block, filters, kernel_size)
        
        x = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_first", padding='same',
                   use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        
        return (x)
        
    def conv_layer(self, x, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, data_format="channels_first", padding='same',
                   use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(self.reg_const))(x)
        
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        
        return (x)
    
    def copy_weights(self, other):
        # copies weights from this neural network to the another network
        for main_layer, other_layer in zip(self.nn.layers, other.nn.layers):
            weights = main_layer.get_weights()
            other_layer.set_weights(weights)
            
    def predict_one(self, board):
        inp = board.reshape(self.num_states)
        return self.nn.predict(np.array(inp, ndmin=4))
    
    def predict_batch(self, boards):
        return self.nn.predict(boards)
    
    def train_batch(self, x_batch, y_batch, batch, epochs=1, use_fit=True):
        if use_fit:
            return self.nn.fit(x_batch, y_batch, verbose=0, epochs=epochs, batch_size=32)
        else:
            return self.nn.train_on_batch(x_batch, y_batch)
    
    def save(self, game, version):
        self.nn.save("./models/version" + "{0:0>4}".format(version) + ".h5")
        
    def plot(self):
        from keras.utils import plot_model
        plot_model(self.nn, to_file="/home/pi/programs/Gobblet_AlphaZero/model_vizualization.png", show_shapes=True, show_layer_names=True)
    
if __name__ == '__main__':
    from Environment import Environment
    env = Environment(4, 4, 4)

    model = Model((4, 4, 4), 16*12, config.HIDDEN_LAYERS, config.REG_CONST, config.LEARNING_RATE)
    model.plot()