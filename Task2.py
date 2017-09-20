import numpy as np
import matplotlib.pyplot as plt
import math
import random
N=200
beta = 2
def get_pattern_value():
    r = random.random()
    if r < 0.5:
        return -1
    else:
        return 1

def get_pattern(p_size):
    return [get_pattern_value() for i in range(p_size)]


def create_random_patterns(nbr_of_patterns, p_size):
    patterns = dict()
    for mu in range(nbr_of_patterns):
        patterns[mu] = get_pattern(p_size)
    return patterns


def calculate_weight(i, j, patterns):
    total_sum = 0
    for mu, bits in patterns.items():
        total_sum += bits[i] * bits[j]
    return total_sum/N


def create_weight_matrix(patterns):
    weight_matrix = [[0 for x in range(N)] for y in range(N)]
    for i in range(N):
        for j in range(i+1):
            if j == i:
                continue
            tmp_weight = calculate_weight(i, j, patterns)
            weight_matrix[i][j] = tmp_weight
            weight_matrix[j][i] = tmp_weight

    return weight_matrix

def calculate_bi(i_index,weights, prev_S):
    sum = 0
    for j in range(N):
        sum += weights[i_index][j]*prev_S[j]
    return sum


def calculate_g(b):
    math.tanh(beta*b)
    return 1/(1+math.exp(-2*beta*b))

def update_m(S, pattern):
    sum = 0
    for j in range(N):
        sum += pattern[j]*S[j]
    #print(sum/N)
    return sum/N

def update_state(g):
    r = random.random()
    if r<g:
        return +1
    else:
        return -1


def main():

    for j in range(20):
        p = 40
        random_patterns = create_random_patterns(p, N)
        weights = create_weight_matrix(random_patterns)

        m_check = 0
        feed_pattern = random_patterns[m_check]
        # S=[get_pattern_value() for i in range(N)]
        S = feed_pattern
        t_max = 1000
        m = dict()
        m[j] = list()
        for i in range(t_max):
            m_value =update_m(S, feed_pattern)
            m[j].append(m_value)

            update_index = random.randint(0, N-1)
            b_update = calculate_bi(update_index, weights, S)

            g_update = calculate_g(b_update)
            tmp = S[update_index]
            #print(b_update*tmp)
            #print(g_update)
            tmp_g =update_state(g_update)
            S[update_index] = tmp_g
            #print("%d,%d,%d"%(tmp, tmp_g, S[update_index]))
    #        print(b_update, g_update, S[update_index])
        m[j].append(update_m(S, feed_pattern))
        plt.plot(range(t_max+1), m[j])
    plt.show()


if __name__ == '__main__':
    main()
