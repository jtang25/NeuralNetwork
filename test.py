import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the data
digits = pd.read_csv('Datasets/digit_train.csv')
digit = np.array([
    np.array([digits.values[0][1:].reshape(28, 28)])
])
label = np.array([
    digits.values[0][0]
])

from Dense_Layer import Dense_Layer
from Output_Layer import Output_Layer
from Flatten import Flatten
from Model import Model

model = Model(Layers=[
    Flatten(),
    Dense_Layer(input_shape=784, output_shape=5, activation_func="relu"),
    Dense_Layer(input_shape=5,output_shape=6, activation_func="relu"),
    Output_Layer(input_shape=6,output_shape=1)
])

model.forward_pass(digit[0])
error = model.fit(digit, label, learning_rate=0.001, epochs=10)
plt.plot(error)