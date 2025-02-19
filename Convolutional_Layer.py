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
        self.dfilters = np.zeros_like(self.filter)  # For storing gradients

    def get_filters(self):
        return self.filter
    
    def get_stride(self):
        return self.stride
    
    def get_padding(self):
        return self.padding
    
    def forward_pass(self, X):
        self.input = X  # Store input for backpropagation
        num_filters = len(self.filter)
        num_images, input_height, input_width = X.shape
        filter_size = self.filter[0].shape[0]
        output_height = ((input_height - filter_size + 2 * self.padding) // self.stride) + 1
        output_width = ((input_width - filter_size + 2 * self.padding) // self.stride) + 1
        output = np.zeros((num_filters, output_height, output_width))

        if self.padding > 0:
            X = np.pad(X, 
                    ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                    mode='constant', constant_values=0)

        for i, image in enumerate(X):
            for h in range(output_height):
                for w in range(output_width):
                    for j, filter in enumerate(self.filter):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + filter_size
                        w_end = w_start + filter_size
                        region = image[h_start:h_end, w_start:w_end]
                        output[j, h, w] += np.sum(region * filter)
        
        output /= num_images
        
        if self.activation_func is not None:
            output = self.activation_func(output)
        self.output = output  # Store output for backpropagation
        return output

    def backward_pass(self, d_output, learning_rate):
        batch_size, num_filters, output_height, output_width = d_output.shape
        _, input_depth, input_height, input_width = self.input.shape
        filter_size = self.filter.shape[2]

        if self.padding > 0:
            padded_input = np.pad(self.input,
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant', constant_values=0)
            d_input = np.zeros_like(padded_input)
        else:
            padded_input = self.input
            d_input = np.zeros_like(self.input)

        # Reset filter gradients for this batch
        self.dfilters = np.zeros_like(self.filter)

        for b in range(batch_size):  # Loop over batch
            for i in range(num_filters):  # Loop over filters
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + filter_size
                        w_end = w_start + filter_size

                        region = padded_input[b, :, h_start:h_end, w_start:w_end]

                        # Update filter gradient
                        self.dfilters[i] += region * d_output[b, i, h, w]

                        # Backpropagate the error to the input
                        d_input[b, :, h_start:h_end, w_start:w_end] += self.filter[i] * d_output[b, i, h, w]

        # Remove padding from d_input if applicable
        if self.padding > 0:
            d_input = d_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # Update filters using gradients
        self.filter -= learning_rate * self.dfilters / batch_size

        return d_input