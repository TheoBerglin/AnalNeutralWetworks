import random
print("Adas äter otvättad röv till frukost")

p = [x for x in range(0, 420, 20)]
p[0] = 1
N = 200
nbr_of_bits = pow(10, 5)


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

"""
def calculate_C(fixed_i, nu, patterns):
    total_sum = 0
    fixed_z = patterns[nu][fixed_i]
    fixed_pattern = patterns[nu]
    for mu, bits in patterns.items():
        if mu == nu:
            continue
        for j in range(0, len(bits)):
            total_sum += fixed_z * bits[fixed_i] * bits[j] * fixed_pattern[j]

    return -total_sum/len(fixed_pattern)
"""


def calculate_weight(i, j, patterns):
    total_sum = 0
    for mu, bits in patterns.items():
        total_sum += bits[i] * bits[j]
    return total_sum/N


def create_weight_matrix(patterns):
    weight_matrix = [[0 for x in range(N)] for y in range(N)]
    for i in range(N):
        for j in range(i+1):
            tmp_weight = calculate_weight(i, j, patterns)
            weight_matrix[i][j] = tmp_weight
            weight_matrix[j][i] = tmp_weight

    return weight_matrix


def calculate_error_prob(nbr_of_patterns):
    bit_counter = 0
    correct_sign_counter = 0
    while bit_counter < nbr_of_bits:
        random_patterns = create_random_patterns(nbr_of_patterns, N)
        weights = create_weight_matrix(random_patterns)
        for nu, bits in random_patterns.items():
            for i in range(N):
                tmp_sum = 0
                for j in range(N):
                    tmp_sum += weights[i][j] * bits[j]
                if tmp_sum == 0:
                    continue
                tmp_sum = 1 if tmp_sum > 0 else -1
                if tmp_sum == bits[i]:
                    correct_sign_counter += 1
                bit_counter += 1

    return correct_sign_counter/nbr_of_bits




print(calculate_error_prob(20))

