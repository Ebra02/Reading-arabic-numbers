import os
import matplotlib.pyplot as plt
import numpy as np
import math
import nnfs  # Assuming this is your neural network library (not included here)

# Define the Layer class
class Layer:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def ReLU(self):
        # Apply ReLU without saving
        return np.maximum(0, self.output)

    def ApplyReLU(self):
        # Apply ReLU and save the results in output
        self.output = np.maximum(0, self.output)

    def ApplySoftMax(self):
        # Apply Softmax activation function
        exp_values = np.exp(self.output - np.max(self.output, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Define the cross-entropy loss function
def cross_entropy_loss(predictions, targets):
    # Avoid numerical instability by adding a small epsilon value
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    # Calculate the cross-entropy loss
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
        c = 0

        for imageFile in dir_list:
            img = plt.imread(path + "//" + imageFile)
            arr = np.array(img, dtype=float)  # Change this line
            flat_arr = arr.ravel()
            X.append(flat_arr)
            y.append(digit)
            c += 1
            if c > 100:
                break

    return np.array(X, dtype=float), np.array(y, dtype=int)  # Change this line

# Define your neural network architecture
def build_neural_network():
    nnfs.init()  # Initialize your neural network library
    X, y = load_arabic_data()  # Load the Arabic numeral dataset

    # Create Dense layers with appropriate input and output sizes
    layer1 = Layer(784, 784)  # Adjust the input size and output size
    layer2 = Layer(784, 10)   # Adjust the input size and output size

    # Define batch size and initial learning rate
    batch_size = 32
    initial_learning_rate = 0.1

    # Training loop
    lowest_loss = float('inf')  # Set an initial high loss value
    best_weights1 = None
    best_biases1 = None
    best_weights2 = None
    best_biases2 = None

    for iteration in range(2000):
        # Shuffle the dataset and split it into mini-batches
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, len(X_shuffled), batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Update weights with small random values
            layer1.weights += 0.05 * np.random.randn(784, 784)
            layer1.biases += 0.05 * np.random.randn(1, 784)
            layer2.weights += 0.05 * np.random.randn(784, 10)
            layer2.biases += 0.05 * np.random.randn(1, 10)

            # Forward pass
            layer1.forward(X_batch)
            layer1.ApplyReLU()
            layer2.forward(layer1.output)
            layer2.ApplySoftMax()

            # Calculate loss
            loss = cross_entropy_loss(layer2.output, y_batch)

            # Calculate accuracy
            accuracy = calculate_accuracy(layer2.output, y_batch)

            # Check if the current model has the lowest loss
            if loss < lowest_loss:
                print('New set of weights found, iteration:', iteration,
                      'loss:', loss, 'acc:', accuracy)
                best_weights1 = layer1.weights.copy()
                best_biases1 = layer1.biases.copy()
                best_weights2 = layer2.weights.copy()
                best_biases2 = layer2.biases.copy()
                lowest_loss = loss
            else:
                # Revert to the best weights and biases found so far
                layer1.weights = best_weights1.copy()
                layer1.biases = best_biases1.copy()
                layer2.weights = best_weights2.copy()
                layer2.biases = best_biases2.copy()

        # Learning rate scheduling (reduce learning rate over time)
        if iteration % 100 == 0:
            initial_learning_rate *= 0.9

# Main function
def main():
    # Load Arabic numeral dataset
    X_arabic, y_arabic = load_arabic_data()

    # Train your neural network
    build_neural_network()

if __name__ == "__main__":
    main()
