import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def create_biases(number_of_neurons):
    bias = np.zeros(number_of_neurons)
    for i in range(number_of_neurons):
        r = random.uniform(-1, 1)
        bias[i] = r
    return bias


def calculate_energy(target, output):
    return 1 / 2 * (target - output) ** 2


def calculate_energy_prime(target, output):
    return target - output


def calculate_b(weights, input_data, bias, neurons_prev_layer, index):
    b = 0
    for i in range(neurons_prev_layer):
        b += weights[i][index] * input_data[i]
    return b - bias


def calculate_activation(b, beta):
    return np.tanh(b * beta)


def calculate_activation_prime(b, beta):
    return beta * (1 - calculate_activation(b, beta) ** 2)


def calculate_weight_update(learning_rate, b, input_data, output, target, beta):
    energy_prime = calculate_energy_prime(target, output)
    activation_prime = calculate_activation_prime(b, beta)
    return learning_rate * energy_prime * activation_prime * input_data


def calculate_first_layer_delta(delta_second_layer, old_weights, b, beta):
    return delta_second_layer * old_weights * calculate_activation_prime(b, beta)


def calculate_bias_update(learning_rate, b, output, target, beta):
    energy_prime = calculate_energy_prime(target, output)
    activation_prime = calculate_activation_prime(b, beta)
    return -learning_rate * energy_prime * activation_prime


def run_3a():
    train_dim1, train_dim2, train_target = read_data('train_data_2017.txt')
    valid_dim1, valid_dim2, valid_target = read_data('valid_data_2017.txt')
    learning_rate = 0.02
    beta = 1 / 2
    class_error_train = np.zeros(10)
    class_error_val = np.zeros(10)
    iterations = 10 ** 6

    for runs in range(10):
        # Initialize weights
        weights = create_weights(2, 1)
        # Initialize bias
        bias = create_biases(1)
        outputs_train_all = np.zeros(300)
        outputs_val_all = np.zeros(200)
        energy_index = 0
        calc_energy_index = 100
        energy_values = np.zeros([2, int(iterations / calc_energy_index)])

        for i in tqdm(iterable=range(iterations), desc="Time steps for run %d" % runs, mininterval=15):
            # Calculate output on train data
            rand = random.randint(0, 299)
            input_data = [train_dim1[rand], train_dim2[rand]]
            target_data = train_target[rand]
            b = calculate_b(weights, input_data, bias, 2, 0)
            output = calculate_activation(b, beta)

            # Calculate gradient and update weights and bias
            for index in range(len(weights)):
                weights[index] += calculate_weight_update(learning_rate, b, input_data[index], output, target_data,
                                                          beta)
            bias += calculate_bias_update(learning_rate, b, output, target_data, beta)

            # Calculate energy values every 100 time step
            if i % calc_energy_index == 0:
                # Calculate energy values for train set
                for index in range(300):
                    input_data = [train_dim1[index], train_dim2[index]]
                    target_data = train_target[index]
                    b = calculate_b(weights, input_data, bias, 2, 0)
                    outputs_train_all[index] = calculate_activation(b, beta)
                    energy_values[0, energy_index] += calculate_energy(target_data, outputs_train_all[index])

                # Calculate energy values for valid set
                for index in range(200):
                    valid_input_data = [valid_dim1[index], valid_dim2[index]]
                    valid_target_data = valid_target[index]
                    b_valid = calculate_b(weights, valid_input_data, bias, 2, 0)
                    outputs_val_all[index] = calculate_activation(b_valid, beta)
                    energy_values[1, energy_index] += calculate_energy(valid_target_data, outputs_val_all[index])
                energy_index += 1  # Keep track of enery index

        # Plot
        plt.figure(1)
        plt.plot(range(0, iterations, calc_energy_index), energy_values[0, :])
        plt.figure(2)
        plt.plot(range(0, iterations, calc_energy_index), energy_values[1, :])

        class_error_train[runs] = sum(abs(train_target - outputs_train_all)) / (2 * 300)
        class_error_val[runs] = sum(abs(valid_target - outputs_val_all)) / (2 * 200)
    # Save the classification errors for training and validation set
    with open('classification_errors_3a.txt', 'w') as file:
        file.write('Train: ')
        file.write(str(class_error_train))
        file.write('\nValid: ')
        file.write(str(class_error_val))
    # Error calculations
    avg_class_error = [np.mean(class_error_train), np.mean(class_error_val)]
    min_class_error = [np.min(class_error_train), np.min(class_error_val)]
    var_class_error = [np.var(class_error_train), np.var(class_error_val)]
    print('Average training classification error: %f \t Average validation classification error: %f \n ' % (
        avg_class_error[0], avg_class_error[1]))
    print('Minimum training classification error: %f \t Minimum validation classification error: %f \n ' % (
        min_class_error[0], min_class_error[1]))
    print('Variance training classification error: %f \t Variance validation classification error: %f \n ' % (
        var_class_error[0], var_class_error[1]))
    # Plotting
    plt.xlabel('Time steps')
    plt.ylabel('Energy function')
    plt.title('Validation set')
    plt.figure(1)
    plt.xlabel('Time steps')
    plt.ylabel('Energy function')
    plt.title('Training set')
    plt.show()


def run_3b():
    # Initialize weights and bias and read train/valid data
    train_dim1, train_dim2, train_target = read_data('train_data_2017.txt')
    valid_dim1, valid_dim2, valid_target = read_data('valid_data_2017.txt')
    learning_rate = 0.02
    beta = 1 / 2
    class_error_train = np.zeros(10)
    class_error_val = np.zeros(10)
    calc_energy_index = 100
    iterations = 10 ** 6
    for runs in range(10):
        # Initialize weights
        weights_first_layer = create_weights(2, 4)
        weights_second_layer = create_weights(4, 1)
        # Initialize biases
        bias_first_layer = create_biases(4)
        bias_second_layer = create_biases(1)
        # Energy calculations
        energy_values = np.zeros([2, int(iterations / calc_energy_index)])
        energy_index = 0
        outputs_train_all = np.zeros(300)
        outputs_val_all = np.zeros(200)

        for i in tqdm(iterable=range(iterations), desc="Timesteps for run %d" % runs, mininterval=15):
            # Calculate output on train data
            rand = random.randint(0, 299)
            input_data = [train_dim1[rand], train_dim2[rand]]
            target_data = train_target[rand]
            b_first_layer = np.zeros(4)
            output_first_layer = np.zeros(4)
            for j in range(4):
                b_first_layer[j] = calculate_b(weights_first_layer, input_data, bias_first_layer[j], 2, j)
                output_first_layer[j] = calculate_activation(b_first_layer[j], beta)
            b_output = calculate_b(weights_second_layer, output_first_layer, bias_second_layer, 4, 0)
            output = calculate_activation(b_output, beta)

            # Update weights and biases
            weights_second_layer_tmp = weights_second_layer
            delta_second_layer = (target_data - output) * calculate_activation_prime(b_output, beta)
            for index in range(len(weights_second_layer)):
                weights_second_layer[index] += calculate_weight_update(learning_rate, b_output,
                                                                       output_first_layer[index],
                                                                       output, target_data, beta)
            bias_second_layer += calculate_bias_update(learning_rate, b_output, output, target_data, beta)

            for column in range(weights_first_layer.shape[1]):
                delta_first_layer_tmp = calculate_first_layer_delta(delta_second_layer,
                                                                    weights_second_layer_tmp[column],
                                                                    b_first_layer[column], beta)
                for row in range(weights_first_layer.shape[0]):
                    weight_update_tmp = delta_first_layer_tmp * learning_rate * input_data[row]
                    weights_first_layer[row][column] += weight_update_tmp
                bias_first_layer[column] += -learning_rate * delta_first_layer_tmp
            if i % calc_energy_index == 0:
                # Calculate energy values for train set
                for index in range(300):
                    input_data = [train_dim1[index], train_dim2[index]]
                    target_data = train_target[index]
                    b_first_layer = np.zeros(4)
                    output_first_layer = np.zeros(4)
                    for j in range(4):
                        b_first_layer[j] = calculate_b(weights_first_layer, input_data, bias_first_layer[j], 2, j)
                        output_first_layer[j] = calculate_activation(b_first_layer[j], beta)
                    b_output = calculate_b(weights_second_layer, output_first_layer, bias_second_layer, 4, 0)
                    outputs_train_all[index] = calculate_activation(b_output, beta)
                    energy_values[0, energy_index] += calculate_energy(target_data, outputs_train_all[index])

                # Calculate energy values for valid set
                for index in range(200):
                    valid_input_data = [valid_dim1[index], valid_dim2[index]]
                    valid_target_data = valid_target[index]
                    b_valid_first_layer = np.zeros(4)
                    output_valid_first_layer = np.zeros(4)
                    for j in range(4):
                        b_valid_first_layer[j] = calculate_b(weights_first_layer, valid_input_data, bias_first_layer[j],
                                                             2, j)
                        output_valid_first_layer[j] = calculate_activation(b_valid_first_layer[j], beta)
                    b_valid_output = calculate_b(weights_second_layer, output_valid_first_layer, bias_second_layer, 4,
                                                 0)
                    outputs_val_all[index] = calculate_activation(b_valid_output, beta)
                    energy_values[1, energy_index] += calculate_energy(valid_target_data, outputs_val_all[index])
                energy_index += 1
        plt.figure(1)
        plt.plot(range(0, iterations, calc_energy_index), energy_values[0, :])
        plt.figure(2)
        plt.plot(range(0, iterations, calc_energy_index), energy_values[1, :])

        class_error_train[runs] = sum(abs(train_target - outputs_train_all)) / (2 * 300)
        class_error_val[runs] = sum(abs(valid_target - outputs_val_all)) / (2 * 200)
    # Save the classification errors for training and validation set
    with open('classification_errors_3b.txt', 'w') as file:
        file.write('Train: ')
        file.write(str(class_error_train))
        file.write('\nValid: ')
        file.write(str(class_error_val))
    # Output calculations
    avg_class_error = [np.mean(class_error_train), np.mean(class_error_val)]
    min_class_error = [np.min(class_error_train), np.min(class_error_val)]
    var_class_error = [np.var(class_error_train), np.var(class_error_val)]
    print('Average training classification error: %f \t Average validation classification error: %f \n ' % (
        avg_class_error[0], avg_class_error[1]))
    print('Minimum training classification error: %f \t Minimum validation classification error: %f \n ' % (
        min_class_error[0], min_class_error[1]))
    print('Variance training classification error: %f \t Variance validation classification error: %f \n ' % (
        var_class_error[0], var_class_error[1]))
    # Plotting
    plt.xlabel('Time steps')
    plt.ylabel('Energy function')
    plt.title('Validation set')
    plt.figure(1)
    plt.xlabel('Time steps')
    plt.ylabel('Energy function')
    plt.title('Training set')
    plt.show()

def plot_training_data():
    train_dim1, train_dim2, train_target = read_data('train_data_2017.txt')
    for i in range(300):
        if train_target[i] == 1:
            plt.scatter(train_dim1[i], train_dim2[i], color='b')
        else:
            plt.scatter(train_dim1[i], train_dim2[i], color='r')
    plt.show()
if __name__ == '__main__':
    plot_training_data()
    #run_3a()
    # run_3b()
