import numpy as np
import matplotlib.pyplot as plt
import random


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
            if j == i:
                continue
            tmp_weight = calculate_weight(i, j, patterns)
            weight_matrix[i][j] = tmp_weight
            weight_matrix[j][i] = tmp_weight
    return weight_matrix


def activation_function(b, beta):
    return 1/(1+np.exp(-2*beta*b))


def local_field(weights, S, index_i):
    sum = 0
    wi = weights[index_i]
    N = len(wi)
    for j in range(N):
        sum += wi[j]*S[j]
    return sum


def state_update(g):
    return 1 if random.random() < g else -1


def calculate_order_parameter(N, S, pattern):
    sum = 0
    for i in range(N):
        sum += S[i]*pattern[i]
    return sum/N


def calculate_m_over_time(N, beta, feed_pattern, weights, time_max):
    m = list()
    S = feed_pattern.copy()
    for t in range(time_max):
        m.append(calculate_order_parameter(N, S, feed_pattern))
        async_update_index = random.randint(0, N-1)
        b_update = local_field(weights, S, async_update_index)
        g_update = activation_function(b_update, beta)
        S[async_update_index] = state_update(g_update)
    return m


if __name__ == '__main__':
    for i in range(20):
        N = 200
        beta = 2
        p = 5
        patterns = create_random_patterns(N, p)
        weights = create_weight_matrix(patterns, N)

        time_max = 10000

        m = calculate_m_over_time(N, beta, patterns[0], weights, time_max)
        plt.plot(range(len(m)), m)

    plt.ylim([0, 1])
    plt.show()
