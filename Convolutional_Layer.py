import numpy as np
from Neuron import *

class Convolutional_Layer:
    def __init__(self, filters, filter_size, stride, padding, activation_func):
        self.filter = []
        for x in range(filters):
            self.filter.append(np.random.uniform(-1, 1, (filter_size, filter_size)))
        self.filter = np.array(self.filter)
        self.stride = stride
        self.padding = padding
        self.activation_func = activation_func
    
    def forward_pass(self, X):
        num_filters = len(self.filter)
        num_images, input_height, input_width = X.shape
        filter_size = self.filter[0].shape[0]
        output_height = ((input_height - filter_size + 2 * self.padding) // self.stride) + 1
        output_width = ((input_width - filter_size + 2 * self.padding) // self.stride) + 1
        output = np.zeros((num_images*num_filters, output_height, output_width))

        if self.padding > 0:
            X = np.pad(X, 
                    ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                    mode='constant', constant_values=0)

        for i, image in enumerate(X):
            for j, filter in enumerate(self.filter):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + filter_size
                        w_end = w_start + filter_size
                        region = image[h_start:h_end, w_start:w_end]
                        output[i * num_filters + j, h, w] = np.sum(region * filter)
        
        # Apply activation function if provided
        if self.activation_func is not None:
            output = self.activation_func(output)
        return output
