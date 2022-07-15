from numpy import dot, exp, random, array

class NeuralNetwork():
    def __init__(neu):
        random.seed(1)
        neu.synaptic_weights = 2 * random.random((4, 1)) - 1

    def __sigmoid(neu, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(neu, x):
        return x * (1 - x)

    def training(neu, inputs, outputs, iterations):
        for iteration in range(iterations):
            output = neu.learn(inputs)
            error = outputs - output
            adjustment = dot(inputs.T, error * neu.__sigmoid_derivative(output)) # transpose matrix
            neu.synaptic_weights += adjustment

    def learn(neu, inputs):
        return neu.__sigmoid(dot(inputs, neu.synaptic_weights))

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    inputs = array([[0, 0, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 1]])
    outputs = array([[0, 1, 1, 0, 1]]).T
    neural_network.training(inputs, outputs, 100000)

    print("Synaptic weights (post training):")
    print(neural_network.synaptic_weights)

    print("New matrix: [1, 1, 0, 0] = ")
    print(neural_network.learn(array([1, 1, 0, 0])))
