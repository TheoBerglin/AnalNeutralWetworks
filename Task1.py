import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import csv


def get_pattern_value():
    r = random.random()
    if r < 0.5:
        return -1
    else:
        return 1


def create_random_patterns(N, p):
    patterns = np.zeros((p, N), dtype=np.int)
    for pattern in range(p):
        for bit in range(N):
            patterns[pattern][bit] = get_pattern_value()
    return patterns


def calculate_weight(i, j, patterns):
    total_sum = 0
    for pattern in patterns:
        total_sum += pattern[i] * pattern[j]
    return total_sum/N


def create_weight_matrix(patterns, N):
    weight_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1):
            tmp_weight = calculate_weight(i, j, patterns)
            weight_matrix[i][j] = tmp_weight
            weight_matrix[j][i] = tmp_weight
    return weight_matrix


def calculate_error_prob(nbr_of_patterns, N, tot_nbr_of_bits):
    bit_counter = 0
    correct_sign_counter = 0
    while bit_counter < tot_nbr_of_bits:
        random_patterns = create_random_patterns(N, nbr_of_patterns)
        weights = create_weight_matrix(random_patterns, N)
        for pattern in random_patterns:
            for bits in range(N):
                tmp_sum = 0
                for j in range(N):
                    tmp_sum += weights[bits][j] * pattern[j]
                if tmp_sum == 0:
                    continue
                correct_sign_counter += 1 if pattern[bits]*tmp_sum > 0 else 0
                bit_counter += 1
                if bit_counter == tot_nbr_of_bits:
                    return 1 - correct_sign_counter / tot_nbr_of_bits


def generate_error_result(number_of_patterns, N, tot_nbr_of_bits):
    error_prob = []
    with open('results.txt', 'w') as file:
        for value in tqdm(number_of_patterns):
            temp_error = calculate_error_prob(value, N, tot_nbr_of_bits)
            error_prob.append(temp_error)
            file.write('%s\t%s\n' % (value, temp_error))


def derived_error_function(N, P):
    alpha = N/P
    return 1/2*(1-np.erf((1+alpha)/math.sqrt(2*alpha)))


def derived_error_function_alpha(alpha):
    return 1/2*(1-math.erf((1+alpha)/math.sqrt(2*alpha)))


def open_result_file():
    x_values_generated = list()
    y_values_generated = list()
    with open('results.txt', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            x_values_generated.append(row[0])
            y_values_generated.append(row[1])
    return x_values_generated, y_values_generated


def plot_main():
    error_function_x = np.linspace(0.01, 2, 400)
    error_function_y = [derived_error_function_alpha(alpha) for alpha in error_function_x]
    plt.plot(error_function_x, error_function_y, label='Theoretical values')
    x_values_generated, y_values_generated = open_result_file()
    x_values_generated = [int(x)/N for x in x_values_generated]
    plt.scatter(x_values_generated, y_values_generated, color='r', label='Numerical simulation')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$P_{error}$')
    plt.legend(loc = 4)
    plt.show()


if __name__ == '__main__':
    p = [x for x in range(0, 420, 20)]
    p[0] = 1
    N = 200
    tot_nbr_of_bits = pow(10, 5)
    #generate_error_result(p, N, tot_nbr_of_bits)
    plot_main()

