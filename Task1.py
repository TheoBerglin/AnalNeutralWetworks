import random
print("Adas äter otvättad röv till frukost")

p = [x for x in range(0, 420, 20)]
p[0] = 1
nbr_of_bits = pow(10, 5)


def get_pattern_value():
    r = random.random()
    if r < 0.5:
        return -1
    else:
        return 1


def get_pattern(p_size):
    return [get_pattern_value() for i in range(0, p_size)]


def create_random_patterns(nbr_of_patterns, p_size):
    patterns = dict()
    for i in range(0, nbr_of_patterns):
        patterns[i] = get_pattern(p_size)
    return patterns

print(create_random_patterns(4, 10))



