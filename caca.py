'''
Created by Kevin De Angeli
Date: 2019-10-26
'''

import matplotlib.pyplot as plt
import random
import numpy as np
#import loader
import pickle
import gzip
from mpl_toolkits import mplot3d

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)
def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        #The len of the list is equal to the number of layers
        self.num_layers = len(sizes)


        self.sizes = sizes

        #This line creates a list with random biases based on the number of
        #neurons in each layer. So for a 784x330x10 it crates a list of [30x1 10x1]
        #print(len(self.biases[1])) #This line shows that idea.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]


        #The function zip returns pairs.
        #if you give it [1,2,3] and [a,b,c] it returns [1,a],[2,b],[3,c]
        #This is perfect to initializing the weights of the matrixes that you we
        #use for feedwards/dot_product operations.
        #Here sizes[:-1] is all numbers except for the last one
        #sizes[1:] is all numbers except for the first one.
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, returnAccuracy = False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)


        trainingError = []
        testError = []

        # For each epoch, go through the entire dataset:
        for j in range(epochs):
            random.shuffle(training_data)

            #Split the data into batches of size mini_batches:
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test));

            else:
                print("Epoch {} complete".format(j))

        ##################################################
        #########Code for Error analysis##################
        ##################################################

            #After each Epoch, Chech the error:
        ##This implements the error for the training data
        #     errorArray = []
        #     for x,y in training_data:
        #         err = .5*np.sum((self.feedforward(x)-y)**2)
        #         errorArray.append(err)
        #     errorAverage =  np.sum(errorArray) #/ (len(errorArray))
        #     print("loss: ", errorAverage)
        #     trainingError.append(errorAverage)
        #
        # #if trainingError == True:
        # x = np.arange(epochs) + 1
        # self.plotError(x,trainingError)


        ##This implements the error for the test data
        ##if trainingError == True:
            #After each Epoch, Chech the error:
            # errorArray = []
            # for x,y in test_data:
            #     err = .5*np.sum((self.feedforward(x)- vectorized_result(y))**2)
            #     errorArray.append(err)
            # errorAverage =  np.sum(errorArray) #/ (len(errorArray))
            # print("loss: ", errorAverage)
            # trainingError.append(errorAverage)
    #
    # #if trainingError == True:
    #     print("enter here ")
    #     x = np.arange(epochs) + 1
    #     self.plotError(x,trainingError)

        if returnAccuracy==True:
            return self.evaluate(test_data)/n_test


    def plotError(self, x, y):
        print("here too")
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
        plt.plot(x, y)
        ax.set(xlabel='Epoch', ylabel='Loss')
        title = ' '
        # plt.legend()
        ax.grid()
        plt.show()




    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #print(mini_batch[0][0]) #is the first picture of the mini_batch
        #print(mini_batch[0][1]) #is the label of the first picture.


        for x, y in mini_batch:
            #Here y is a 1x10 vector.
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer

        i = 1
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #print(activations[2])#Contains the output of the NN

        #You need to accumulate the error for each mini_batch, then avrage it over the mini_batch
        #Then average all the errors of all the mini_batches. That's the error per batch.
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    #I created a speicial function for XOR because the old "Evaluate" function
    #would not work with this tiny dataset.
    #Here, I compute the accuracy manually so it can be plooted
    #in the function XOR.
    def XOR_SGD(self, training_data, epochs, mini_batch_size, eta,
                test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)


        dataSet = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]])
        right = -1
        guess = -1
        accuracy = []
        acc = 0

        # For each epoch, go through the entire dataset:
        for j in range(epochs):
            random.shuffle(training_data)

            # Split the data into batches of size mini_batches:
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test));

            else:
                print("Epoch {} complete".format(j))

            acc=0
            for i in range(4):
                predict = self.feedforward(np.reshape((dataSet[0][i]), (2, 1)))
                actual = dataSet[1][i]
                if predict[0] > predict[1]:
                    guess = 0
                else:
                    guess = 1

                if guess == actual:
                    acc += 1
            accuracy.append(acc / 4)

        return accuracy


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


def Neurons2DGrpah(training_data, test_data):
    #let's start by changing the nubmer of neurons
    neuronNum = np.arange(1,50)
    accuracy = []
    test = list(test_data)
    for neuron in neuronNum:
        #test = copy.copy(test_data)
        print("Number of Neurons: ", neuron)
        net = Network([784, neuron, 10])
        accuracy.append(net.SGD(training_data, 2, 10, 3.0, test_data=test,returnAccuracy= True))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(neuronNum, accuracy)
    ax.set(xlabel='Number of Neurons', ylabel='Accuracy')
    title=' '
    #plt.legend()
    ax.grid()
    plt.show()
    print(accuracy)

def EpoachAccuracy2D(training_data, test_data):
    #let's start by changing the nubmer of neurons
    EpochNums = np.arange(1,51)
    accuracy = []
    test = list(test_data)
    for epochN in EpochNums:
        print("Number of Epochs: ", epochN)
        net = Network([784, 30, 10])
        accuracy.append(net.SGD(training_data, epochN, 10, 3.0, test_data=test,returnAccuracy= True))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(EpochNums, accuracy)
    ax.set(xlabel='Number of Epochs', ylabel='Accuracy')
    title=' '
    #plt.legend()
    ax.grid()
    plt.show()
    print(accuracy)

def NeuronsVSLayersVsAccuracy3D(training_data, test_data):
    #Note: The array of neurons and the array of hl should be the same
    #The program can be modified so it can take arbitrary numbers.
    neurons = np.arange(1, 11)  # Controls number of neurons in all layers.
    hl = np.arange(1,11)  # Controls number of layers
    network = []
    network.append(784)
    test = list(test_data)
    network.append(-1)
    accuracy =  np.zeros([hl.shape[0],neurons.shape[0]])
    for i in hl:
        network.append(10)
        for n in neurons:
            for k in range(len(network) - 2):
                network[k + 1] = n
            print("Network Architecture Being used: ",network)
            net = Network(network)
            acc = net.SGD(training_data, 1, 10, 3.0, test_data=test, returnAccuracy=True)
            acc = np.round_(acc,decimals=2)
            accuracy[hl.shape[0]-i][n-1] = acc

    print(accuracy)

    #Note: The heat map fuction displays the graph in the exact order as the
    #Matrix. So I had to populate the matrix in the opposite order
    left = neurons[0] - .5  # Should be set so that it starts a the first point in the array -.5
    right = neurons[-1] + .5  # last number of the array +.5
    bottom = hl[0] - .5
    top = hl[-1] + .5
    extent = [left, right, bottom, top]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    im = ax.imshow(accuracy, extent=extent, interpolation='nearest')
    ax.set(xlabel='Number of Neurons', ylabel='Number of Layers')

    #Label each square in the heat map:
    for i in range(len(hl)):
        for j in range(len(neurons)):
            text = ax.text(j + 1, i + 1, accuracy[accuracy.shape[0]-i-1,j],
                           ha="center", va="center", color="w")

    plt.show()

def LearningRateAnalysis(training_data, test_data):
    lr_array = np.linspace(0.1,20,40)
    accuracy = []
    test = list(test_data)
    for lr in lr_array:
        #test = copy.copy(test_data)
        print("Learning Rate: ", lr)
        net = Network([784, 30, 10])
        acc = net.SGD(training_data, 1, 10, lr, test_data=test,returnAccuracy= True)
        accuracy.append(acc)
    print("Accuracy Array: ", accuracy)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(lr_array, accuracy)
    ax.set(xlabel='Learning Rate', ylabel='Accuracy')
    title=' '
    #plt.legend()
    ax.grid()
    plt.show()
    print(accuracy)

def LayersAccuracy2D(training_data, test_data):
    hl = np.arange(1, 11)  # Controls number of layers
    network = []
    accuracy = []
    test=list(test_data)
    network.append(784)
    network.append(10)
    for i in hl:
        network.insert(1, 30)
        net = Network(network)
        acc = net.SGD(training_data, 1, 10, 3.0, test_data=test, returnAccuracy=True)
        accuracy.append(acc)
        print("Network Architecture Being used: ", network)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(hl, accuracy)
    ax.set(xlabel='Number of Layers', ylabel='Accuracy')
    title=' '
    #plt.legend()
    ax.grid()
    plt.show()
    print(accuracy)

def XORvectorized_result(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e


def XOR():
    EpochNums = 100
    data= np.array([[0.,0.], [1.,0.], [0.,1.],[1.,1.]])
    label= np.array([0,1,1,0])
    training_input = [np.reshape(x,(2,1)) for x in data]
    training_labels= [XORvectorized_result(y) for y in label]
    training_data =  list(zip(training_input,training_labels))

    net = Network([2, 10, 2])
    accuracyArr = net.XOR_SGD(training_data, EpochNums, 1, 2, test_data=None)


    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(np.arange(EpochNums), accuracyArr)
    ax.set(xlabel='Epoch', ylabel='Accuracy')
    title=' '
    #plt.legend()
    ax.grid()
    plt.show()



    # x= np.array([1,0])
    # x = np.reshape(x,(2,1))
    # #print(net.feedforward(x))


def main():
    training_data, validation_data, test_data = load_data_wrapper()
    training_data = list(training_data)


    net = Network([784, 30, 10])
    print(len(net.weights[0][0]))
    print(len(net.weights[1][0]))


    #net = Network([784, 30, 10])
    #net.SGD(training_data, 30, 10, 3.0, test_data=list(test_data),returnAccuracy=False)


    #The numbers above represent the number of neurons per each layer. The size of the array is the number of layers.

    #net.SGD(training_data, 30, 10, 3.0, test_data=list(test_data))

    #In order they appear, each number represent: Num of epochs, Mini Batch Size, learning rate.
    #The original was 30,10,3


    #Neurons2DGrpah(training_data=training_data,test_data=test_data)
    #EpoachAccuracy2D(training_data=training_data,test_data=test_data)
    #NeuronsVSLayersVsAccuracy3D(training_data=training_data,test_data=test_data)
    #LearningRateAnalysis(training_data=training_data,test_data=test_data)
    #LayersAccuracy2D(training_data=training_data,test_data=test_data)


    #XOR()




if __name__ == "__main__":
    main()
