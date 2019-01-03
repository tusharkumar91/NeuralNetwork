import numpy as np
from collections import defaultdict

from com.uofr.course.csc442.hw.hw4.models.BaseModel import BaseModel


class NeuralNetwork(BaseModel):
    r"""
    Neural network implementation using sigmoid activations
    and cross entropy loss.
    Please Note that this implementation would only work
    for classification tasks and not for predicting
    continuous attributes.
    Also the data must be a categorical data because
    we convert into their one hot encodings.
    """
    def __init__(self, layer_sizes, attribute_file, target_attribute,
                 train_data_file=None, test_data_file=None, train_data=None, test_data=None):
        BaseModel.__init__(self, attribute_file=attribute_file, target_attribute=target_attribute,
                           train_data_file=train_data_file, test_data_file=test_data_file, train_data=train_data,
                           test_data=test_data)
        self.attribute_vector_map = {}
        self.num_of_layers = len(layer_sizes)
        self.layers = layer_sizes
        self.biases = [np.random.randn(layer_size, 1) for layer_size in layer_sizes[1:]]
        self.weights = [np.random.randn(layerj, layeri) for layeri, layerj in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.test_data_vectorized = False

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    @staticmethod
    def cross_entropy_cost(output, y):
        cost = np.sum(np.nan_to_num(-y * np.log(output) - (1 - y) * np.log(1 - output)))
        return cost

    @staticmethod
    def cross_entropy_cost_derivative(output, y):
        return output - y

    def forward(self, x):
        out = x
        for bias, weights in zip(self.biases, self.weights):
            out = self.sigmoid(np.dot(weights, out) + bias)
        return out

    def convert_attribute_one_hot_vectors(self):
        r"""
        Converts all attribute values to their one hot encoding values
        and saves them for use during converting dataset
        to one hot vectors
        :return: None
        """
        attribute_vector_map = defaultdict(dict)
        for attribute in self.attribute_names:
            attribute_values = self.attribute_domains[attribute]
            for idx, value in enumerate(attribute_values):
                attribute_vector = np.zeros((len(attribute_values)), dtype=int)
                attribute_vector[idx] = 1
                attribute_vector_map[attribute][value] = attribute_vector
        self.attribute_vector_map = attribute_vector_map

    def convert_data_to_one_hot_vector(self, data):
        r"""
        Converts data to the form where each row is now a
        concatenated vector of one hot encoded values of
        its attributes
        :param data: data to be converted
        :return: vectorized data
        """
        vectorized_data = []
        for row in data:
            vectorized_row = []
            output = None
            for attribute in self.attribute_names:
                if attribute == self.target_attribute:
                    output = self.attribute_vector_map[attribute][row[attribute]].\
                        reshape(len(self.attribute_vector_map[attribute].keys()), 1)
                    continue
                attribute_vector = self.attribute_vector_map[attribute][row[attribute]]
                vectorized_row.extend(attribute_vector)
            vectorized_data.append((np.array(vectorized_row).reshape(-1,1), output))
        return vectorized_data

    def train(self, batch_size=10, epochs=100, learning_rate=0.01, validation_interval=10):
        r"""
        Method to train the neural network using mini batch and stochiastic gradient descent.
        :param batch_size: Batch size
        :param epochs:
        :param learning_rate:
        :param validation_interval:
        :return:
        """

        self.get_attribute_data()
        self.get_train_data()

        self.convert_attribute_one_hot_vectors()
        if self.train_data is None and self.train_data_file is None:
                raise ValueError("Neither training data not training file provided")

        self.train_data = self.convert_data_to_one_hot_vector(data=self.train_data)

        self.sgd(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
                 validation_interval=validation_interval)

    def predict(self):
        r"""
        Method to test neural network by executing the feed forward on
        the test data input and finding the predicted label.
        Please note that the returned value is a tuple of
        predicted value and actual value
        """
        self.get_test_data()
        if not self.test_data_vectorized:
            self.test_data = self.convert_data_to_one_hot_vector(data=self.test_data)
            self.test_data_vectorized = True
        pred = [(np.argmax(self.forward(x).reshape(1, -1), axis=1), np.argmax(y, axis=0))
                        for (x, y) in self.test_data]
        return pred

    def sgd(self, batch_size, epochs, learning_rate, validation_interval):
        r"""
        Method to run stochiastic gradient descent using a mini batch technique.
        :param batch_size: batch size for mini batch
        :param epochs: NUmber epochs to train for
        :param learning_rate:
        :param validation_interval: interval at which the test data should be tested
            and accuracy should be reported.
        :return:
        """
        """losses = []
        accuracy = []
        iters = []
        iter = 1
        count=1
        cost = 0.0
        for x, y in self.train_data:
            output = self.forward(x)
            cost_n = self.cross_entropy_cost(output, y)
            cost += cost_n / len(self.train_data)
        losses.append(cost)
        iters.append(count)
        count += 1
        predictions = self.predict()
        accuracy.append(self.get_accuracy(predictions=predictions))"""

        for epoch in range(1, epochs+1):
            mini_batch = [self.train_data[k:k + batch_size] for k in range(0, len(self.train_data), batch_size)]
            for data in mini_batch:
                self.update_mini_batch(data, learning_rate)
                """iter += 1
                if iter % 15000 == 0:
                    print("Iteration {}".format(count))
                    cost = 0.0
                    for x, y in self.train_data:
                        output = self.forward(x)
                        cost_n = self.cross_entropy_cost(output, y)
                        cost += cost_n / len(self.train_data)
                    losses.append(cost)
                    iters.append(count)
                    count += 1
                    predictions = self.predict()
                    accuracy.append(self.get_accuracy(predictions=predictions))"""
            print("Epoch {} complete".format(epoch))
            cost = 0.0

            if epoch % validation_interval == 0:
                for x, y in self.train_data:
                    output = self.forward(x)
                    cost_n = self.cross_entropy_cost(output, y)
                    cost += cost_n / len(self.train_data)
                print("Cost on training data: {}".format(cost))
                predictions = self.predict()
                print("Testing Accuracy : {} ".format(self.get_accuracy(predictions=predictions)))
                print()
        """print(losses)
        print(accuracy)
        print(iters)"""

    def update_mini_batch(self, data, learning_rate):
        r"""
        Meat of SGD method which runs back propagation on the data
        and updates the bias and weights with the gradient for this
        mini batch
        :param data:
        :param learning_rate:
        :return:
        """
        delta_bias = [np.zeros(bias.shape) for bias in self.biases]
        delta_weights = [np.zeros(weights.shape) for weights in self.weights]
        for x, y in data:
            delta_bias_for_row, delta_weights_for_row = self.back_propagation(x, y)
            delta_bias = [del_bias + new_del_bias for del_bias, new_del_bias in zip(delta_bias, delta_bias_for_row)]
            delta_weights = [del_weights + new_del_weights for del_weights, new_del_weights in
                            zip(delta_weights, delta_weights_for_row)]
        self.biases = [b - ((learning_rate / len(data)) * delB) for b, delB in zip(self.biases, delta_bias)]
        self.weights = [w - ((learning_rate / len(data)) * delW) for w, delW in zip(self.weights, delta_weights)]

    def back_propagation(self, x, y):
        r"""
        Runs the feed foreward process for the input and then
        back propagates teh error using cross entropy loss
        :param x: input
        :param y: output
        :return:
        """
        delta_biases = [np.zeros(bias.shape) for bias in self.biases]
        delta_weights = [np.zeros(weight.shape) for weight in self.weights]
        activation = x
        activations = [x]
        Z = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            Z.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.cross_entropy_cost_derivative(activations[-1], y) * self.sigmoid_derivative(Z[-1])
        delta_biases[-1] = delta
        delta_weights[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_of_layers):
            z = Z[-layer]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            delta_biases[-layer] = delta
            delta_weights[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return delta_biases, delta_weights

    def get_accuracy(self, predictions):
        r"""
        Method to get accuracy of the learned model on test data
        by checking for equality of predicted label and test label
        :param predictions:
        :return:
        """
        correct = sum(int(x == y) for (x, y) in predictions)
        return float(correct)/len(predictions) * 100
