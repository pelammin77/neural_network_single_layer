"""
file: Neural_network.py
Author: Petri Lamminaho
email: lammpe77@gmail.com
"""

from numpy import random, array, exp, dot


class Neural_network():
    def __init__(self):
        random.seed(1)  # antaa ainna samat numerot kun ohjelma käy
        self.synaptic_weights = 2 * random.random((3, 1)) - 1  # luo neutronin jolla on
        # kolme inputtia ja antaa yhden(1) outputi

    def __sigmoid(self, x):
        """
        private function
        sigmoid function pass the data
         and normalize data to 1 or 0
        :param x:
        :return: 1 or 0
        """
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        """
        private function
        :param x:
        :return derivative function :
        """
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):

             for iteration in range(number_of_training_iterations):
                 output = self.think(training_set_inputs)

                 error = training_set_outputs - output
                 #print("error:", error)
#
                 adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

                 # Adjust the weights.
                 self.synaptic_weights += adjustment


    def think(self, inputs):
        """
        network thinks
        :param inputs:
        :return: sigmoid function
        pass the data (inputs) through  the network
        single neutron
        """
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    nn = Neural_network()
    print("random start weights")
    print(nn.synaptic_weights)

    training_data_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_data_outputs = array([[0, 1, 1, 0]]).T

    print("Training data inputs:")
    print(training_data_inputs)
    print("Training data outputs:")
    print(training_data_outputs)
    nn.train(training_data_inputs, training_data_outputs, 10000)
    print("New weights after training: ")
    print(nn.synaptic_weights)

    # Test the neural network with a new situation.
    print ("Tring new input data [1, 0, 0 ] -> ?: ( output should be 1) ")
    print(nn.think(array([1, 0, 0])))







