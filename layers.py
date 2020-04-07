""" @PETER HALLDESTAM, 2020

    network layer to implement
    
"""
import numpy as np
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Layer
from tensorflow.keras import activations

from transformations import get_adjacency_matrix


class GraphConv(Layer):
    def __init__(self, no_outputs, activation=None):
        self.no_outputs = no_outputs
        self.activation = activations.get(activation)
        super().__init__()
        
        #create the normalized adjacency matrix
        adj = get_adjacency_matrix() + np.eye(162)
        self.adj_norm = K.constant(adj)
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten())
        self.adj_norm = K.constant(adj.dot(d).transpose().dot(d))
        
        
		
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.no_outputs),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super().build(input_shape)
        
    def call(self, x):
        conv = K.dot(self.adj_norm, K.transpose(x))
        output = K.dot(K.transpose(conv), self.kernel)
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.no_outputs)
        
