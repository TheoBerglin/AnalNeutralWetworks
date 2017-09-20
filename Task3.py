import numpy as np
import random
import csv

print('Theo äter infekterad varfylld pungkula till middag')


def read_data(filename):
    file = open(filename, 'r')
    patterns_dim1 = []
    patterns_dim2 = []
    target_output = []
    spamreader = csv.reader(file, delimiter='\t')
    for line in spamreader:
        patterns_dim1.append(float(line[0]))
        patterns_dim2.append(float(line[1]))
        target_output.append(float(line[2]))
    patterns_dim1 = (patterns_dim1 - np.mean(patterns_dim1)) / np.std(patterns_dim1)
    patterns_dim2 = (patterns_dim2 - np.mean(patterns_dim2)) / np.std(patterns_dim2)
    return [patterns_dim1, patterns_dim2, target_output]


def create_weights(input_size, target_size):
    weights = np.zeros([input_size, target_size])
    for i in range(input_size):
        for j in range(target_size):
            r = random.uniform(-0.2, 0.2)
            weights[i][j] = r
    return weights


def create_biases(number_of_layers):
    bias = np.zeros(number_of_layers)
    for i in range(number_of_layers):
        r = random.uniform(-1, 1)
        bias[i] = r
    return bias


def calculate_energy(target, output):
    return 1/2 * (target - output)**2


def calculate_energy_prime(target, output):
    return (target - output)


def calculate_b(weights, input_data, bias, neurons_prev_layer, index):
    sum = 0
    for i in range(neurons_prev_layer):
        sum += weights[i][index-1] * input_data[i]
    return sum - bias


def calculate_activation(b, beta):
    return np.tanh(b*beta)


def calculate_activation_prime(b, beta):
    return 1 - calculate_activation(b, beta)**2


def calculate_weight_update(learning_rate, b, input_data, output, target, beta):
    energy_prime = calculate_energy_prime(target, output)
    activation_prime = calculate_activation_prime(b, beta)
    return learning_rate*energy_prime*activation_prime*input_data


def main():
    train_dim1, train_dim2, train_target = read_data('train_data_2017.txt')
    valid_dim1, valid_dim2, valid_target = read_data('valid_data_2017.txt')
    learning_rate = 0.02
    beta = 1/2
    weights = create_weights(2, 1)
    bias = create_biases(1)
    for i in range(10**6):
        rand = random.randint(0, 299)
        input_data = [train_dim1[rand], train_dim2[rand]]
        b = calculate_b(weights, input_data, bias, 2, 1)
        output = calculate_activation(b, beta)
        print(weights)
        # loopa över lager och neuroner
        # loopa över bias, hur många?
        for index in range(len(weights)):
            weights[index] += calculate_weight_update(learning_rate, b, input_data[index], output, train_target[rand], beta)
        print(weights)


if __name__ == '__main__':
    main()