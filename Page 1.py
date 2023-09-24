import os
import matplotlib.pyplot as plt
import numpy as np
import math
import nnfs
from sklearn.utils import shuffle

# Set the random seed for reproducibility
np.random.seed(42)
nnfs.init()

# Define the Layer class
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def ReLU(self):
        return np.maximum(0, self.output)

    def ApplyReLU(self):
        self.output = np.maximum(0, self.output)

    def ApplySoftMax(self):
        exp_values = np.exp(self.output - np.max(self.output, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Define the cross-entropy loss function
def cross_entropy_loss(predictions, targets):
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(np.log(predictions[np.arange(N), targets])) / N
    return ce_loss

# Define the accuracy calculation function
def calculate_accuracy(predictions, targets):
    predicted_labels = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_labels == targets)
    total_predictions = len(targets)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Data loading function
def load_arabic_data():
    path_init = r"C:\Users\Ebra\OneDrive\سطح المكتب\Projects\Reading-arabic-numbers//SmallDataset"
    y = []
    X = []

    for digit in range(10):
        path = path_init + "//" + str(digit)
        dir_list = os.listdir(path)

        for imageFile in dir_list:
            img = plt.imread(path + "//" + imageFile)
            arr = np.array(img, dtype=float)
            flat_arr = arr.ravel()
            X.append(flat_arr)
            y.append(digit)

    return np.array(X, dtype=float), np.array(y, dtype=int)

# Define your neural network architecture
def build_neural_network():
    X, y = load_arabic_data()

    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)

    # Create Dense layers with appropriate input and output sizes
    layer1 = Layer(784, 784)
    layer2 = Layer(784, 10)

    # Define batch size and initial learning rate
    batch_size = 32
    initial_learning_rate = 0.005

    # Training loop
    lowest_loss = float('inf')
    best_weights1 = None
    best_biases1 = None
    best_weights2 = None
    best_biases2 = None

    for iteration in range(2000):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            # Set the random seed before updating weights
            np.random.seed(iteration + i)

            layer1.weights += initial_learning_rate * np.random.randn(784, 784)
            layer1.biases += initial_learning_rate * np.random.randn(1, 784)
            layer2.weights += initial_learning_rate * np.random.randn(784, 10)
            layer2.biases += initial_learning_rate * np.random.randn(1, 10)

            layer1.forward(X_batch)
            layer1.ApplyReLU()
            layer2.forward(layer1.output)
            layer2.ApplySoftMax()

            loss = cross_entropy_loss(layer2.output, y_batch)
            accuracy = calculate_accuracy(layer2.output, y_batch)

            if loss < lowest_loss:
                print('New set of weights found, iteration:', iteration,
                      'loss:', loss, 'acc:', accuracy)
                best_weights1 = layer1.weights.copy()
                best_biases1 = layer1.biases.copy()
                best_weights2 = layer2.weights.copy()
                best_biases2 = layer2.biases.copy()
                lowest_loss = loss
            else:
                layer1.weights = best_weights1.copy()
                layer1.biases = best_biases1.copy()
                layer2.weights = best_weights2.copy()
                layer2.biases = best_biases2.copy()

        if iteration % 100 == 0:
            initial_learning_rate *= 0.9

# Main function
def main():
    build_neural_network()

if __name__ == "__main__":
    main()
