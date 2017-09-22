import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


def get_pattern_value():
    r = random.random()
    if r < 0.5:
        return -1
    else:
        return 1


def create_random_patterns(nbr_of_neurons, p):
    patterns = np.zeros((p, nbr_of_neurons), dtype=np.int)
    for pattern in range(p):
        for bit in range(nbr_of_neurons):
            patterns[pattern][bit] = get_pattern_value()
    return patterns


def calculate_weight(i, j, patterns):
    weight = 0
    for pattern in patterns:
        weight += pattern[i] * pattern[j]
    return weight / N


def create_weight_matrix(patterns, nbr_of_neurons):
    weight_matrix = np.zeros((nbr_of_neurons, nbr_of_neurons))
    for i in range(nbr_of_neurons):
        for j in range(i + 1):
            if j == i:
                continue
            tmp_weight = calculate_weight(i, j, patterns)
            weight_matrix[i][j] = tmp_weight
            weight_matrix[j][i] = tmp_weight
    return weight_matrix


def activation_function(b, beta):
    return 1 / (1 + np.exp(-2 * beta * b))


def local_field(weights, S, index_i):
    b = 0
    wi = weights[index_i]
    N = len(wi)
    for j in range(N):
        b += wi[j] * S[j]
    return b


def state_update(g):
    return 1 if random.random() < g else -1


def calculate_order_parameter(nbr_of_neurons, states, pattern):
    order_parameter = 0
    for i in range(nbr_of_neurons):
        order_parameter += states[i] * pattern[i]
    return order_parameter / nbr_of_neurons


def calculate_m_over_time(nbr_of_neurons, beta, feed_pattern, weights, time_max):
    m = list()
    S = feed_pattern.copy()
    m_mean = list()
    for t in tqdm(iterable=range(time_max), mininterval=15):
        m_t = calculate_order_parameter(nbr_of_neurons, S, feed_pattern)
        m.append(m_t)
        if len(m_mean) == 0:
            m_mean.append(m_t)
        else:
            m_mean.append((m_mean[-1] * (t - 1) + m_t) / t)
        async_update_index = random.randint(0, N - 1)
        b_update = local_field(weights, S, async_update_index)
        g_update = activation_function(b_update, beta)
        S[async_update_index] = state_update(g_update)
    return m, m_mean


if __name__ == '__main__':
    N = 200
    beta = 2
    p = 5
    for i in range(20):
        patterns = create_random_patterns(N, p)
        weights = create_weight_matrix(patterns, N)
        time_max = 10 ** 6
        m, m_mean = calculate_m_over_time(N, beta, patterns[0], weights, time_max)
        plt.plot(range(len(m_mean)), m_mean)
    plt.ylim([0, 1])
    plt.xlabel('Time steps')
    plt.ylabel('Order parameter')
    plt.show()
