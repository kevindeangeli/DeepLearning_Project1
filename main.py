'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-01-15

'''

import numpy as np
import matplotlib.pyplot as plt

class Neuron():

    def __init__(self,inputLen, activationFun = "sigmoid", learningRate = .1, weights = None):
        self.inputLen = inputLen
        self.learnR = learningRate
        self.activationFunction = activationFun

        if weights is None:
            #set them to random:
            self.weights = [np.random.random_sample() for i in range(inputLen)]


    def activate(self):
        '''
        given a value returns its value after activation(depending on the activation function used).
        :return:
        '''
        if self.activationFunction == "sigmoid":
            a=1
            print(a)

    def calculate(self, input):
        '''
        Given an input, it will calculate the output
        :return:
        '''
        return np.dot(input, self.weights)


class FullyConnectedLayer():
    def __init__(self, inputLen, numOfNeurons = 5, activationFun = "sigmoid", learningRate = .1, weights = None):
        self.inputLen = inputLen
        self.neuronsNum = numOfNeurons
        self.activationFun = activationFun
        self.learningRate = learningRate
        self.weights = weights
        self.neurons = [Neuron(inputLen=self.inputLen, activationFun=activationFun, learningRate=self.learningRate, weights=self.weights) for i in range(numOfNeurons)]

    def calculate(self, input):
        '''
        Will calculate the output of all the neurons in the layer.
        :return:
        '''
        layerOutput = []
        for neuron in self.neurons:
            layerOutput.append(neuron.calculate(input))

        return layerOutput


class NeuralNetwork():
    def __init__(self, inputLen, layersNum = 2, neuronsNum = None, activationVector = 0, lossFunction = "MSE", learningRate = .1, weights = None):
        self.inputLen   = inputLen
        self.layersNum  = layersNum
        self.activationVector = activationVector
        self.lossFunction = lossFunction
        self.learningRate = learningRate
        self.weights = weights

        if neuronsNum is None :  #By default, each layer will have 5 neurons, unless specified.
            self.neuronsNum = [3 for i in range(layersNum)]
        else:
            self.neuronsNum = neuronsNum

        if activationVector is None or activationVector != layersNum: #This is the default vector if a problem is encountered or the vector is not provided when the class is created.
            self.activationVector = ["sigmoid" for i in range(layersNum)]

        #Define the layers of the networks with the respective neurons:
        self.layers = []
        inputLenLayer = inputLen
        for i in range(self.layersNum):
            self.layers.append(FullyConnectedLayer(numOfNeurons = self.neuronsNum[i], activationFun = self.activationVector[i],inputLen=inputLenLayer, learningRate=self.learningRate, weights=self.weights))
            #The number of weights in one layer depends on the number of neurons in the previous layer:
            inputLenLayer = self.neuronsNum[i]


        self.showWeights()

    def showWeights(self):
        #Function which just goes through each neuron in each layer and displays the weights.
        inputLenLayer = self.inputLen
        for i in range(self.layersNum):
            print(" ")
            for k in range(self.neuronsNum[i]):
                print(self.layers[i].neurons[k].weights)

            inputLenLayer = self.neuronsNum[i]





    def calculate(self, input):
        '''
        given an input calculates the output of the network.
        input should be a list.
        :return:
        '''
        output = input
        for layer in self.layers:
            output = layer.calculate(output)


        return output




    def calculateLoss(self):
        x=0

    def train(self):
        '''
        Given a single input and desired output, it will take one step of gradient descent.
        :return:
        '''
        x=0




def main():

    #The number of neurons in the last layers should be equal to the number of classes.
    #there are easier ways to do this, but I'm just following the instructions.
    model = NeuralNetwork(inputLen=2,layersNum=2, neuronsNum=[3,2])
    model.showWeights()
    out= model.calculate([1,0])
    print(out)

if __name__ == "__main__":
    main()
