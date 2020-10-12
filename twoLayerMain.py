import numpy as np
import sys, os

from neuralNetworkClass import Two_Layer

neuralNetwork = Two_Layer(4, 10, 3)
neuralNetwork.learn(0.1, 10000)
